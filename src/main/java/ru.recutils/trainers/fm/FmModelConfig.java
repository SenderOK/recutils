package ru.recutils.trainers.fm;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;

public class FmModelConfig implements Serializable {
    public final int dimension;
    public final float featureWeightsRegularizer;
    public final float embeddingsRegularizer;

    public FmModelConfig(int dimension, float featureWeightsRegularizer, float embeddingsRegularizer) {
        this.dimension = dimension;
        this.featureWeightsRegularizer = featureWeightsRegularizer;
        this.embeddingsRegularizer = embeddingsRegularizer;
    }

    public static FmModelConfig fromCommandLineArguments(CommandLineArguments args) {
        return new FmModelConfig(args.dimension, args.featureWeightsRegularizer, args.embeddingsRegularizer);
    }
}
