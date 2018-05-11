package ru.recutils.trainers.regression;

import java.util.Map;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import ru.recutils.common.ObservationHolder;
import ru.recutils.lossfuncs.LossFunction;
import ru.recutils.trainers.SgdTrainerConfig;

class RegressionSgdTrainer {
    private final SgdTrainerConfig trainerConfig;
    private final Random randomGen;
    private final ForkJoinPool forkJoinPool;
    private final LossFunction lossFunction;

    public RegressionSgdTrainer(SgdTrainerConfig trainerConfig) {
        this.trainerConfig = trainerConfig;
        this.randomGen = new Random(trainerConfig.seed);
        this.forkJoinPool = new ForkJoinPool(trainerConfig.numThreads);
        this.lossFunction = trainerConfig.lossFunctionType.getLossFunction();
    }

    <T extends ObservationHolder> boolean train(
            Iterable<T> dataset,
            RegressionModelWeights regressionModelWeights,
            RegressionModelConfig modelConfig)
    {
        for (int i = 0; i < trainerConfig.numIter; ++i) {
            System.out.println("training epoch #" + i);

            DoubleAdder lossSum = new DoubleAdder();
            AtomicInteger objectCount = new AtomicInteger(0);
            try {
                Stream<T> dataStream = StreamSupport.stream(dataset.spliterator(), true);
                if (!dataStream.iterator().hasNext()) {
                    return false;
                }
                forkJoinPool.submit(() -> dataStream.forEach(observation -> {
                    if (observation == null) {
                        return;
                    }

                    float loss = updateModelWeightsAndReturnLoss(observation, regressionModelWeights, modelConfig);

                    lossSum.add(loss);
                    objectCount.incrementAndGet();
                })).get();
            } catch (Exception e) {
                System.err.println("Failed to train in parallel");
                return false;
            }

            if (objectCount.get() > 0) {
                System.out.println("Average loss on " + objectCount + " objects is "
                        + lossSum.sum() / objectCount.get());
            } else {
                System.out.println("No objects processed");
            }
        }
        return true;
    }

    private <T extends ObservationHolder> float updateModelWeightsAndReturnLoss(
            T observation,
            RegressionModelWeights regressionModelWeights,
            RegressionModelConfig modelConfig)
    {
        // random initialization for new weights
        for (Integer featureHash : observation.getFeatures().keySet()) {
            if (!regressionModelWeights.featureWeights.containsKey(featureHash)) {
                regressionModelWeights.featureWeights.put(
                        featureHash, (float)randomGen.nextGaussian() * trainerConfig.initStddev);
            }
        }

        float prediction = regressionModelWeights.apply(observation);
        float label = observation.getLabel();
        float importance = observation.getImportance();
        float loss = lossFunction.value(prediction, label);
        float dLdp = lossFunction.derivative(prediction, label);

        // updating bias
        regressionModelWeights.bias -= trainerConfig.learningRate * dLdp; // 2 atomic ops, but nobody cares

        // updating weights
        for (Map.Entry<Integer, Float> entry : observation.getFeatures().entrySet()) {
            int featureHash = entry.getKey();
            float featureValue = entry.getValue();
            float gradient = dLdp * featureValue + modelConfig.featureWeightsRegularizer
                    * regressionModelWeights.featureWeights.get(featureHash);
            regressionModelWeights.featureWeights.merge(
                    featureHash,
                    -trainerConfig.learningRate * importance * gradient,
                    (a, b) -> a + b
            );
        }
        return loss;
    }
}
