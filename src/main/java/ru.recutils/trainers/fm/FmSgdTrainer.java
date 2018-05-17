package ru.recutils.trainers.fm;

import java.util.Map;

import ru.recutils.common.MathUtils;
import ru.recutils.common.ObservationHolder;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.trainers.SgdTrainer;
import ru.recutils.trainers.SgdTrainerConfig;

public class FmSgdTrainer<T extends ObservationHolder> extends SgdTrainer<T, FmModelWeights, FmModelConfig> {

    FmSgdTrainer(
            SgdTrainerConfig trainerConfig,
            StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter)
    {
        super(trainerConfig, stringToFeaturesHolderConverter);
    }

    @Override
    public float updateModelWeightsAndReturnLoss(T observation, FmModelWeights modelWeights, FmModelConfig modelConfig) {
        int embeddingSize = modelConfig.dimension;
        modelWeights.initializeZeroWeights(observation, randomGen, trainerConfig.initStddev);

        float prediction = modelWeights.apply(observation);
        float label = observation.getLabel();
        float importance = observation.getImportance();
        float loss = lossFunction.value(prediction, label);
        float dLdp = lossFunction.derivative(prediction, label);

        // updating bias
        modelWeights.regressionModelWeights.bias -= currentLearningRate *
                lossFunction.derivative(prediction, label); // 2 atomic ops, but nobody cares

        // precalculating weighted sum of embedding vectors
        float[] weightedEmbeddingsSum = new float[embeddingSize];
        for (Map.Entry<Integer, Float> entry : observation.getFeatures().entrySet()) {
            int featureHash = entry.getKey();
            float featureValue = entry.getValue();

            MathUtils.inplaceAddWithScale(
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
                    -currentLearningRate * importance * weightGradient,
                    (a, b) -> a + b
            );

            // updating embedding
            float[] embeddingToUpdate = modelWeights.featureEmbeddings.get(featureHash).clone();
            float[] dLde = weightedEmbeddingsSum.clone();
            MathUtils.inplaceAddWithScale(dLde, embeddingToUpdate, -featureValue, embeddingSize);
            MathUtils.inplaceScale(dLde, dLdp * featureValue);
            MathUtils.inplaceAddWithScale(dLde, embeddingToUpdate, modelConfig.embeddingsRegularizer, embeddingSize);
            MathUtils.inplaceScale(dLde, -currentLearningRate * importance);
            modelWeights.featureEmbeddings.merge(featureHash, dLde, (a, b) -> MathUtils.add(a, b));
        }

        return loss;
    }
}
