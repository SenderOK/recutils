package ru.recutils.common;

public enum OptimizationAlgorithmType {
    SGD("sgd"),
    ALS("als")
    ;

    private String name;

    OptimizationAlgorithmType(String name) {
        this.name = name;
    }

    public String getName() {
        return this.name;
    }

    public static OptimizationAlgorithmType fromString(String optimizationAlgorithm) {
        for (OptimizationAlgorithmType algorithm : OptimizationAlgorithmType.values()) {
            if (optimizationAlgorithm.equalsIgnoreCase(algorithm.name)) {
                return algorithm;
            }
        }
        return null;
    }
}
