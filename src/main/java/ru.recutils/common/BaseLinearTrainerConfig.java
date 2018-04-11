package ru.recutils.common;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;

public class BaseLinearTrainerConfig implements Serializable {
    public final double featureWeightsRegularizer;
    public final int numIter;
    public final int numThreads;

    public BaseLinearTrainerConfig(
            double featureWeightsRegularizer,
            int numIter,
            int numThreads)
    {
        this.featureWeightsRegularizer = featureWeightsRegularizer;
        this.numIter = numIter;
        this.numThreads = numThreads;
    }

    public BaseLinearTrainerConfig(BaseLinearTrainerConfig config) {
        this(config.featureWeightsRegularizer, config.numIter, config.numThreads);
    }

    public static BaseLinearTrainerConfig fromCommandLineArguments(CommandLineArguments args) {
        return new BaseLinearTrainerConfig(args.featureWeightsRegularizer, args.numIter, args.numThreads);
    }
}
