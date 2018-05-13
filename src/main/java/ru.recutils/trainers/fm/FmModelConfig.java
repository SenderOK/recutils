package ru.recutils.trainers.fm;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;

public class FmModelConfig implements Serializable {
    final int dimension;
    final float featureWeightsRegularizer;
    final float embeddingsRegularizer;

    private FmModelConfig(int dimension, float featureWeightsRegularizer, float embeddingsRegularizer) {
        this.dimension = dimension;
        this.featureWeightsRegularizer = featureWeightsRegularizer;
        this.embeddingsRegularizer = embeddingsRegularizer;
    }

    public static FmModelConfig fromCommandLineArguments(CommandLineArguments args) {
        return new FmModelConfig(args.dimension, args.featureWeightsRegularizer, args.embeddingsRegularizer);
    }
}
