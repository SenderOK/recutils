package ru.recutils.trainers.fm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import ru.recutils.common.LinearModelWeights;
import ru.recutils.common.Utils;
import ru.recutils.common.ObservationHolder;
import ru.recutils.trainers.regression.RegressionModelWeights;

public class FmModelWeights implements LinearModelWeights, Serializable {
    final RegressionModelWeights regressionModelWeights;
    final HashMap<Integer, float[]> featureEmbeddings;
    final int embeddingsSize;

    public FmModelWeights(int embeddingsSize) {
        this.regressionModelWeights = new RegressionModelWeights();
        this.featureEmbeddings = new HashMap<>();
        this.embeddingsSize = embeddingsSize;
    }

    @Override
    public float apply(ObservationHolder observationHolder) {
        float result = regressionModelWeights.apply(observationHolder);

        Set<Integer> featureSet = observationHolder.getFeatures().keySet();
        List<Integer> features =  new ArrayList<Integer>(featureSet);
        for (int i = 0; i < features.size(); ++i) {
            if (!featureEmbeddings.containsKey(features.get(i))) {
                continue;
            }
            float[] firstEmbedding = featureEmbeddings.get(features.get(i));
            for (int j = i + 1; j < features.size(); ++j) {
                if (!featureEmbeddings.containsKey(features.get(j))) {
                    continue;
                }
                float[] secondEmbedding = featureEmbeddings.get(features.get(j));
                result += Utils.dotProduct(firstEmbedding, secondEmbedding, embeddingsSize);
            }
        }
        return result;
    }
}
