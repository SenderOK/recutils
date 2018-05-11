package ru.recutils.trainers;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;
import ru.recutils.common.OptimizationAlgorithmType;

public class BaseLinearTrainerConfig implements Serializable {
    public final long seed;
    public final float initStddev;
    public final int numIter;

    public BaseLinearTrainerConfig(long seed, float initStddev, int numIter) {
        this.seed = seed;
        this.initStddev = initStddev;
        this.numIter = numIter;
    }

    public BaseLinearTrainerConfig(BaseLinearTrainerConfig config) {
        this(config.seed, config.initStddev, config.numIter);
    }

    public static BaseLinearTrainerConfig fromCommandLineArguments(CommandLineArguments args) {
        return new BaseLinearTrainerConfig(args.seed, args.initStddev, args.numIter);
    }

    public OptimizationAlgorithmType getOptimizationType() {
        return OptimizationAlgorithmType.ALS;
    }
}
