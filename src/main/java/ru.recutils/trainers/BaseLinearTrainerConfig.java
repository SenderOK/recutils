package ru.recutils.trainers;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;
import ru.recutils.common.OptimizationAlgorithmType;

public class BaseLinearTrainerConfig implements Serializable {
    public final long seed;
    public final float initStddev;
    public final int numIter;
    public final boolean useHoldout;

    private BaseLinearTrainerConfig(long seed, float initStddev, int numIter, boolean useHoldout) {
        this.seed = seed;
        this.initStddev = initStddev;
        this.numIter = numIter;
        this.useHoldout = useHoldout;
    }

    BaseLinearTrainerConfig(BaseLinearTrainerConfig config) {
        this(config.seed, config.initStddev, config.numIter, config.useHoldout);
    }

    public static BaseLinearTrainerConfig fromCommandLineArguments(CommandLineArguments args) {
        return new BaseLinearTrainerConfig(args.seed, args.initStddev, args.numIter, args.useHoldout);
    }

    public OptimizationAlgorithmType getOptimizationType() {
        return OptimizationAlgorithmType.ALS;
    }
}
