package ru.recutils.common;

import java.util.Map;

public interface ObservationHolder {
    Map<Integer, Float> getFeatures();

    float getLabel();

    float getImportance();
}
