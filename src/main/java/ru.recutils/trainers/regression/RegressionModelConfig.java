package ru.recutils.trainers.regression;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;

public class RegressionModelConfig implements Serializable {
    public final float featureWeightsRegularizer;

    public RegressionModelConfig(float featureWeightsRegularizer) {
        this.featureWeightsRegularizer = featureWeightsRegularizer;
    }

    public static RegressionModelConfig fromCommandLineArguments(CommandLineArguments args) {
        return new RegressionModelConfig(args.featureWeightsRegularizer);
    }
}
