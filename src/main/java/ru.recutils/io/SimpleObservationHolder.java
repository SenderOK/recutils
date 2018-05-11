package ru.recutils.io;

import java.util.Map;

import ru.recutils.common.ObservationHolder;

public class SimpleObservationHolder implements ObservationHolder {
    private final Map<Integer, Float> featureWeights;
    private final float label;
    private final float importance;

    @Override
    public Map<Integer, Float> getFeatures() {
        return featureWeights;
    }

    @Override
    public float getLabel() {
        return label;
    }

    @Override
    public float getImportance() {
        return importance;
    }

    public SimpleObservationHolder(Map<Integer, Float> featureWeights, float label, float importance) {
        this.featureWeights = featureWeights;
        this.label = label;
        this.importance = importance;
    }
}
