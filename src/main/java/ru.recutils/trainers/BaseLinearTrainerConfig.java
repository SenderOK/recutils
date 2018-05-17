package ru.recutils.trainers;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;
import ru.recutils.common.OptimizationAlgorithmType;

public class BaseLinearTrainerConfig implements Serializable {
    public final long seed;
    public final float initStddev;
    public final int numIter;
    public final boolean useHoldout;
    public final int earlyStoppingIters;

    private BaseLinearTrainerConfig(
            long seed, float initStddev, int numIter, boolean useHoldout, int earlyStoppingIters)
    {
        this.seed = seed;
        this.initStddev = initStddev;
        this.numIter = numIter;
        this.useHoldout = useHoldout;
        this.earlyStoppingIters = earlyStoppingIters;
    }

    BaseLinearTrainerConfig(BaseLinearTrainerConfig config) {
        this(config.seed, config.initStddev, config.numIter, config.useHoldout, config.earlyStoppingIters);
    }

    public static BaseLinearTrainerConfig fromCommandLineArguments(CommandLineArguments args) {
        return new BaseLinearTrainerConfig(args.seed, args.initStddev, args.numIter, args.useHoldout,
                args.earlyStoppingIters);
    }

    public OptimizationAlgorithmType getOptimizationType() {
        return OptimizationAlgorithmType.ALS;
    }
}
