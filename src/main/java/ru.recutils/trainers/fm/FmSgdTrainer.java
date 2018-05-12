package ru.recutils.trainers.fm;

import java.util.Map;

import ru.recutils.common.Utils;
import ru.recutils.common.ObservationHolder;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.trainers.SgdTrainer;
import ru.recutils.trainers.SgdTrainerConfig;

public class FmSgdTrainer<T extends ObservationHolder> extends SgdTrainer<T, FmModelWeights, FmModelConfig> {

    public FmSgdTrainer(
            SgdTrainerConfig trainerConfig,
            StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter)
    {
        super(trainerConfig, stringToFeaturesHolderConverter);
    }

    @Override
    public float updateModelWeightsAndReturnLoss(T observation, FmModelWeights modelWeights, FmModelConfig modelConfig) {
        int embeddingSize = modelConfig.dimension;

        // random initialization for new weights
        for (Integer featureHash : observation.getFeatures().keySet()) {
            modelWeights.regressionModelWeights.featureWeights.putIfAbsent(
                    featureHash, (float)randomGen.nextGaussian() * trainerConfig.initStddev);
            modelWeights.featureEmbeddings.putIfAbsent(featureHash, Utils.getRandomGaussianArray(randomGen,
                        trainerConfig.initStddev, embeddingSize));
        }

        float prediction = modelWeights.apply(observation);
        float label = observation.getLabel();
        float importance = observation.getImportance();
        float loss = lossFunction.value(prediction, label);
        float dLdp = lossFunction.derivative(prediction, label);

        // updating bias
        modelWeights.regressionModelWeights.bias -= trainerConfig.learningRate *
                lossFunction.derivative(prediction, label); // 2 atomic ops, but nobody cares

        // precalculating weighted sum of embedding vectors
        float[] weightedEmbeddingsSum = new float[embeddingSize];
        for (Map.Entry<Integer, Float> entry : observation.getFeatures().entrySet()) {
            int featureHash = entry.getKey();
            float featureValue = entry.getValue();

            Utils.inplaceAddWithScale(
                    weightedEmbeddingsSum,
                    modelWeights.featureEmbeddings.get(featureHash),
                    featureValue,
                    embeddingSize
            );
        }

        for (Map.Entry<Integer, Float> entry : observation.getFeatures().entrySet()) {
            int featureHash = entry.getKey();
            float featureValue = entry.getValue();

            // updating weight
            float weightGradient = dLdp * featureValue + modelConfig.featureWeightsRegularizer
                    * modelWeights.regressionModelWeights.featureWeights.get(featureHash);
            modelWeights.regressionModelWeights.featureWeights.merge(
                    featureHash,
                    -trainerConfig.learningRate * importance * weightGradient,
                    (a, b) -> a + b
            );

            // updating embedding
            float[] embeddingToUpdate = modelWeights.featureEmbeddings.get(featureHash).clone();
            float[] dLde = weightedEmbeddingsSum.clone();
            Utils.inplaceAddWithScale(dLde, embeddingToUpdate, -featureValue, embeddingSize);
            Utils.inplaceScale(dLde, dLdp * featureValue);
            Utils.inplaceAddWithScale(dLde, embeddingToUpdate, modelConfig.embeddingsRegularizer,
                    embeddingSize);
            Utils.inplaceScale(dLde, -trainerConfig.learningRate * importance);

            modelWeights.featureEmbeddings.merge(
                    featureHash,
                    dLde,
                    (a, b) -> Utils.add(a, b)
            );
        }

        return loss;
    }
}
