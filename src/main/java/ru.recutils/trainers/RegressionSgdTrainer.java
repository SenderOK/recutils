package ru.recutils.trainers;

import java.util.Map;
import java.util.Random;

import ru.recutils.common.ObservationHolder;
import ru.recutils.lossfuncs.LossFunction;

public class RegressionSgdTrainer {
    public static <T extends ObservationHolder> void train(
            Iterable<T> dataset,
            RegressionModelWeights regressionModelWeights,
            RegressionModelConfig modelConfig,
            SgdTrainerConfig trainerConfig)
    {
        Random randomGen = new Random(trainerConfig.seed);

        LossFunction lossFunction = trainerConfig.lossFunctionType.getLossFunction();
        for (int i = 0; i < trainerConfig.numIter; ++i) {
            System.out.println("training epoch #" + i);

            double lossSum = 0;
            int objectCount = 0;
            for (T observation : dataset) {
                // random initialization for new weights
                for (Integer featureHash : observation.getFeatures().keySet()) {
                    if (!regressionModelWeights.featureWeights.containsKey(featureHash)) {
                        regressionModelWeights.featureWeights.put(
                                featureHash, randomGen.nextGaussian() * trainerConfig.initStddev);
                    }
                }

                double prediction = regressionModelWeights.apply(observation);
                double label = observation.getLabel();
                double importance = observation.getImportance();
                double dLdp = lossFunction.derivative(prediction, label);

                lossSum += lossFunction.value(prediction, label);
                ++objectCount;

                // updating bias
                regressionModelWeights.bias -= trainerConfig.learningRate * dLdp;

                // updating weights
                for (Map.Entry<Integer, Double> entry : observation.getFeatures().entrySet()) {
                    int featureHash = entry.getKey();
                    double featureValue = entry.getValue();
                    double gradient = dLdp * featureValue + modelConfig.featureWeightsRegularizer
                            * regressionModelWeights.featureWeights.get(featureHash);
                    regressionModelWeights.featureWeights.merge(
                            featureHash,
                            -trainerConfig.learningRate * importance * gradient,
                            (a, b) -> a + b
                    );
                }
            }

            System.out.println("Average loss on " + objectCount + " objects is " + lossSum / objectCount);
        }
    }
}
