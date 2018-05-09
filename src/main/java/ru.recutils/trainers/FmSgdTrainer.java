package ru.recutils.trainers;

import java.util.Map;
import java.util.Random;

import ru.recutils.common.MathUtils;
import ru.recutils.common.ObservationHolder;
import ru.recutils.lossfuncs.LossFunction;

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

            double lossSum = 0;
            int objectCount = 0;
            for (T observation : dataset) {
                // random initialization for new weights
                for (Integer featureHash : observation.getFeatures().keySet()) {
                    if (!fmModelWeights.regressionModelWeights.featureWeights.containsKey(featureHash)) {
                        fmModelWeights.regressionModelWeights.featureWeights.put(
                                featureHash, randomGen.nextGaussian() * trainerConfig.initStddev);
                        fmModelWeights.featureEmbeddings.put(featureHash, MathUtils.getRandomGaussianArray(randomGen,
                                trainerConfig.initStddev, embeddingSize));
                    }
                }

                double prediction = fmModelWeights.apply(observation);
                double label = observation.getLabel();
                double importance = observation.getImportance();
                double dLdp = lossFunction.derivative(prediction, label);

                lossSum += lossFunction.value(prediction, label);
                ++objectCount;

                // updating bias
                fmModelWeights.regressionModelWeights.bias -= trainerConfig.learningRate *
                        lossFunction.derivative(prediction, label);

                // precalculating weighted sum of embedding vectors
                double[] weightedEmbeddingsSum = new double[embeddingSize];
                for (Map.Entry<Integer, Double> entry : observation.getFeatures().entrySet()) {
                    int featureHash = entry.getKey();
                    double featureValue = entry.getValue();

                    MathUtils.inplaceAddWithScale(
                            weightedEmbeddingsSum,
                            fmModelWeights.featureEmbeddings.get(featureHash),
                            featureValue,
                            embeddingSize
                    );
                }

                for (Map.Entry<Integer, Double> entry : observation.getFeatures().entrySet()) {
                    int featureHash = entry.getKey();
                    double featureValue = entry.getValue();

                    // updating weight
                    double weightGradient = dLdp * featureValue + modelConfig.featureWeightsRegularizer
                            * fmModelWeights.regressionModelWeights.featureWeights.get(featureHash);
                    fmModelWeights.regressionModelWeights.featureWeights.merge(
                            featureHash,
                            -trainerConfig.learningRate * importance * weightGradient,
                            (a, b) -> a + b
                    );

                    // updating embedding
                    double[] embeddingToUpdate = fmModelWeights.featureEmbeddings.get(featureHash);
                    double[] dLde = weightedEmbeddingsSum.clone();
                    MathUtils.inplaceAddWithScale(dLde, embeddingToUpdate, -featureValue, embeddingSize);
                    MathUtils.inplaceScale(dLde, dLdp * featureValue);
                    MathUtils.inplaceAddWithScale(dLde, embeddingToUpdate, modelConfig.embeddingsRegularizer,
                            embeddingSize);
                    MathUtils.inplaceAddWithScale(embeddingToUpdate, dLde, -trainerConfig.learningRate * importance,
                            embeddingSize);
                }
            }

            System.out.println("Average loss on " + objectCount + " objects is " + lossSum / objectCount);
        }
    }
}
