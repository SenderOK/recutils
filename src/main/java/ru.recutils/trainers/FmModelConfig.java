package ru.recutils.trainers;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;

public class FmModelConfig implements Serializable {
    public final int dimension;
    public final double featureWeightsRegularizer;
    public final double embeddingsRegularizer;

    public FmModelConfig(int dimension, double featureWeightsRegularizer, double embeddingsRegularizer) {
        this.dimension = dimension;
        this.featureWeightsRegularizer = featureWeightsRegularizer;
        this.embeddingsRegularizer = embeddingsRegularizer;
    }

    public static FmModelConfig fromCommandLineArguments(CommandLineArguments args) {
        return new FmModelConfig(args.dimension, args.featureWeightsRegularizer, args.embeddingsRegularizer);
    }
}
