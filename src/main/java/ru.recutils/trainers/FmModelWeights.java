package ru.recutils.trainers;

import java.io.Serializable;
import java.util.HashMap;

public class FmModelWeights implements Serializable {
    private final HashMap<Integer, Double> featureWeights;
    private final HashMap<Integer, double[]> featureEmbeddings;
    private double bias;

    public FmModelWeights() {
        this.featureWeights = new HashMap<>();
        this.featureEmbeddings = new HashMap<>();
        this.bias = 0;
    }
}
