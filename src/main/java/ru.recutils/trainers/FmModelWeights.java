package ru.recutils.trainers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import ru.recutils.common.MathUtils;
import ru.recutils.common.ObservationHolder;

public class FmModelWeights implements Serializable {
    final RegressionModelWeights regressionModelWeights;
    final HashMap<Integer, double[]> featureEmbeddings;
    final int embeddingsSize;

    public FmModelWeights(int embeddingsSize) {
        this.regressionModelWeights = new RegressionModelWeights();
        this.featureEmbeddings = new HashMap<>();
        this.embeddingsSize = embeddingsSize;
    }

    public double apply(ObservationHolder observationHolder) {
        double result = regressionModelWeights.apply(observationHolder);

        Set<Integer> featureSet = observationHolder.getFeatures().keySet();
        List<Integer> features =  new ArrayList<Integer>(featureSet);
        for (int i = 0; i < features.size(); ++i) {
            if (!featureEmbeddings.containsKey(features.get(i))) {
                continue;
            }
            double[] firstEmbedding = featureEmbeddings.get(features.get(i));
            for (int j = i + 1; j < features.size(); ++j) {
                if (!featureEmbeddings.containsKey(features.get(j))) {
                    continue;
                }
                double[] secondEmbedding = featureEmbeddings.get(features.get(j));
                result += MathUtils.dotProduct(firstEmbedding, secondEmbedding, embeddingsSize);
            }
        }
        return result;
    }
}
