package ru.recutils.trainers.regression;

import java.util.Map;

import ru.recutils.common.ObservationHolder;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.trainers.SgdTrainer;
import ru.recutils.trainers.SgdTrainerConfig;

public class RegressionSgdTrainer<T extends ObservationHolder>
        extends SgdTrainer<T, RegressionModelWeights, RegressionModelConfig> {

    public RegressionSgdTrainer(
            SgdTrainerConfig trainerConfig,
            StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter)
    {
        super(trainerConfig, stringToFeaturesHolderConverter);
    }

    @Override
    public float updateModelWeightsAndReturnLoss(
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
        regressionModelWeights.bias -= currentLearningRate * dLdp; // 2 atomic ops, but nobody cares

        // updating weights
        for (Map.Entry<Integer, Float> entry : observation.getFeatures().entrySet()) {
            int featureHash = entry.getKey();
            float featureValue = entry.getValue();
            float gradient = dLdp * featureValue + modelConfig.featureWeightsRegularizer
                    * regressionModelWeights.featureWeights.get(featureHash);
            regressionModelWeights.featureWeights.merge(
                    featureHash,
                    -currentLearningRate * importance * gradient,
                    (a, b) -> a + b
            );
        }
        return loss;
    }

}
