package ru.recutils.io;

import java.io.Serializable;

public interface FeatureNameHasher extends Serializable {
    int getHash(String featureName);
}
