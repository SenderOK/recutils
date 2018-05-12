package ru.recutils.trainers.fm;

import java.io.Serializable;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import ru.recutils.common.LinearModelWeights;
import ru.recutils.common.Utils;
import ru.recutils.common.ObservationHolder;
import ru.recutils.trainers.regression.RegressionModelWeights;

public class FmModelWeights implements LinearModelWeights, Serializable {
    final RegressionModelWeights regressionModelWeights;
    final ConcurrentHashMap<Integer, float[]> featureEmbeddings;
    final int embeddingsSize;

    public FmModelWeights(int embeddingsSize) {
        this.regressionModelWeights = new RegressionModelWeights();
        this.featureEmbeddings = new ConcurrentHashMap<>();
        this.embeddingsSize = embeddingsSize;
    }

    @Override
    public float apply(ObservationHolder observationHolder) {
        float result = regressionModelWeights.apply(observationHolder);

        float[] linearCombination = new float[embeddingsSize];
        float squaredNormsSum = 0.0f;
        for (Map.Entry<Integer, Float> entry : observationHolder.getFeatures().entrySet()) {
            int featureHash = entry.getKey();
            float featureValue = entry.getValue();
            if (!featureEmbeddings.containsKey(featureHash)) {
                continue;
            }
            float[] embedding = featureEmbeddings.get(featureHash).clone();
            Utils.inplaceAddWithScale(linearCombination, embedding, featureValue, embeddingsSize);
            squaredNormsSum += Utils.l2normSquared(embedding) * featureValue * featureValue;
        }
        return result + 0.5f * (Utils.l2normSquared(linearCombination) - squaredNormsSum);
    }
}
