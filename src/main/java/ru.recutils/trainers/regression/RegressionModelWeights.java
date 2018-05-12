package ru.recutils.trainers.regression;

import java.io.Serializable;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import ru.recutils.common.LinearModelWeights;
import ru.recutils.common.ObservationHolder;

public class RegressionModelWeights implements LinearModelWeights, Serializable {
    public final ConcurrentHashMap<Integer, Float> featureWeights;
    public float bias;

    public RegressionModelWeights() {
        this.featureWeights = new ConcurrentHashMap<>();
        this.bias = 0.0f;
    }

    @Override
    public float apply(ObservationHolder observationHolder) {
        Map<Integer, Float> scanMap = observationHolder.getFeatures();
        Map<Integer, Float> lookupMap = featureWeights;
        if (scanMap.size() > lookupMap.size()) {
            scanMap = featureWeights;
            lookupMap = observationHolder.getFeatures();
        }

        float result = bias;
        for (Map.Entry<Integer, Float> entry : scanMap.entrySet()) {
            result += lookupMap.getOrDefault(entry.getKey(), 0.0f) * entry.getValue();
        }
        return result;
    }
}
