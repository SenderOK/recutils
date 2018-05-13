package ru.recutils.trainers.regression;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;

public class RegressionModelConfig implements Serializable {
    final float featureWeightsRegularizer;

    private RegressionModelConfig(float featureWeightsRegularizer) {
        this.featureWeightsRegularizer = featureWeightsRegularizer;
    }

    public static RegressionModelConfig fromCommandLineArguments(CommandLineArguments args) {
        return new RegressionModelConfig(args.featureWeightsRegularizer);
    }
}
