package ru.recutils.io;

import java.util.Map;

import ru.recutils.common.ObservationHolder;

public class SimpleObservationHolder implements ObservationHolder {
    private final Map<Integer, Double> featureWeights;
    private final double label;
    private final double importance;

    @Override
    public Map<Integer, Double> getFeatures() {
        return featureWeights;
    }

    @Override
    public double getLabel() {
        return label;
    }

    @Override
    public double getImportance() {
        return importance;
    }

    public SimpleObservationHolder(Map<Integer, Double> featureWeights, double label, double importance) {
        this.featureWeights = featureWeights;
        this.label = label;
        this.importance = importance;
    }
}
