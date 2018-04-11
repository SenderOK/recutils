package ru.recutils.common;

import java.util.Map;

public interface ObservationHolder {
    Map<Integer, Double> getFeatures();

    double getLabel();

    double getImportance();
}
