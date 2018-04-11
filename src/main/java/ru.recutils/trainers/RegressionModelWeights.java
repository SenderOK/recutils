package ru.recutils.trainers;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import ru.recutils.common.ObservationHolder;

public class RegressionModelWeights implements Serializable {
    final HashMap<Integer, Double> featureWeights;
    double bias;

    public RegressionModelWeights() {
        this.featureWeights = new HashMap<>();
        this.bias = 0;
    }

    public double apply(ObservationHolder observationHolder) {
        Map<Integer, Double> scanMap = observationHolder.getFeatures();
        Map<Integer, Double> lookupMap = featureWeights;
        if (scanMap.size() > lookupMap.size()) {
            scanMap = featureWeights;
            lookupMap = observationHolder.getFeatures();
        }

        double result = bias;
        for (Map.Entry<Integer, Double> entry : scanMap.entrySet()) {
            result += lookupMap.getOrDefault(entry.getKey(), 0.0) * entry.getValue();
        }
        return result;
    }
}
