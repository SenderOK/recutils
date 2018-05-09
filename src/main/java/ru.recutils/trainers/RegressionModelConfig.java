package ru.recutils.trainers;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;

public class RegressionModelConfig implements Serializable {
    public final double featureWeightsRegularizer;

    public RegressionModelConfig(double featureWeightsRegularizer) {
        this.featureWeightsRegularizer = featureWeightsRegularizer;
    }

    public static RegressionModelConfig fromCommandLineArguments(CommandLineArguments args) {
        return new RegressionModelConfig(args.featureWeightsRegularizer);
    }
}
