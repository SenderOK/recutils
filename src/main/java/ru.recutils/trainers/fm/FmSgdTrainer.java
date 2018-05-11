package ru.recutils.trainers.fm;

import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;

import ru.recutils.common.MathUtils;
import ru.recutils.common.ObservationHolder;
import ru.recutils.lossfuncs.LossFunction;
import ru.recutils.trainers.SgdTrainerConfig;

public class FmSgdTrainer {
    public static <T extends ObservationHolder> void train(
            Iterable<T> dataset,
            FmModelWeights fmModelWeights,
            FmModelConfig modelConfig,
            SgdTrainerConfig trainerConfig)
    {
        Random randomGen = new Random(trainerConfig.seed);
        LossFunction lossFunction = trainerConfig.lossFunctionType.getLossFunction();
        int embeddingSize = modelConfig.dimension;

        for (int iter = 0; iter < trainerConfig.numIter; ++iter) {
            System.out.println("training epoch #" + iter);

            DoubleAdder lossSum = new DoubleAdder();
            AtomicInteger objectCount = new AtomicInteger(0);
            for (T observation : dataset) {
                // random initialization for new weights
                for (Integer featureHash : observation.getFeatures().keySet()) {
                    if (!fmModelWeights.regressionModelWeights.featureWeights.containsKey(featureHash)) {
                        fmModelWeights.regressionModelWeights.featureWeights.put(
                                featureHash, (float)randomGen.nextGaussian() * trainerConfig.initStddev);
                        fmModelWeights.featureEmbeddings.put(featureHash, MathUtils.getRandomGaussianArray(randomGen,
                                trainerConfig.initStddev, embeddingSize));
                    }
                }

                float prediction = fmModelWeights.apply(observation);
                float label = observation.getLabel();
                float importance = observation.getImportance();
                float dLdp = lossFunction.derivative(prediction, label);

                lossSum.add(lossFunction.value(prediction, label));
                objectCount.incrementAndGet();

                // updating bias
                fmModelWeights.regressionModelWeights.bias -= trainerConfig.learningRate *
                        lossFunction.derivative(prediction, label);

                // precalculating weighted sum of embedding vectors
                float[] weightedEmbeddingsSum = new float[embeddingSize];
                for (Map.Entry<Integer, Float> entry : observation.getFeatures().entrySet()) {
                    int featureHash = entry.getKey();
                    float featureValue = entry.getValue();

                    MathUtils.inplaceAddWithScale(
                            weightedEmbeddingsSum,
                            fmModelWeights.featureEmbeddings.get(featureHash),
                            featureValue,
                            embeddingSize
                    );
                }

                for (Map.Entry<Integer, Float> entry : observation.getFeatures().entrySet()) {
                    int featureHash = entry.getKey();
                    float featureValue = entry.getValue();

                    // updating weight
                    float weightGradient = dLdp * featureValue + modelConfig.featureWeightsRegularizer
                            * fmModelWeights.regressionModelWeights.featureWeights.get(featureHash);
                    fmModelWeights.regressionModelWeights.featureWeights.merge(
                            featureHash,
                            -trainerConfig.learningRate * importance * weightGradient,
                            (a, b) -> a + b
                    );

                    // updating embedding
                    float[] embeddingToUpdate = fmModelWeights.featureEmbeddings.get(featureHash);
                    float[] dLde = weightedEmbeddingsSum.clone();
                    MathUtils.inplaceAddWithScale(dLde, embeddingToUpdate, -featureValue, embeddingSize);
                    MathUtils.inplaceScale(dLde, dLdp * featureValue);
                    MathUtils.inplaceAddWithScale(dLde, embeddingToUpdate, modelConfig.embeddingsRegularizer,
                            embeddingSize);
                    MathUtils.inplaceAddWithScale(embeddingToUpdate, dLde, -trainerConfig.learningRate * importance,
                            embeddingSize);
                }
            }

            if (objectCount.get() > 0) {
                System.out.println("Average loss on " + objectCount + " objects is "
                        + lossSum.sum() / objectCount.get());
            } else {
                System.out.println("No objects processed");
            }
        }
    }
}
