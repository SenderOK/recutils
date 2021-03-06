package ru.recutils.trainers;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;
import ru.recutils.common.LossFunctionType;
import ru.recutils.common.OptimizationAlgorithmType;

public class SgdTrainerConfig extends BaseLinearTrainerConfig implements Serializable {
    public final LossFunctionType lossFunctionType;
    public final float learningRate;
    public final float learningRateDecay;
    public final int numThreads;

    private SgdTrainerConfig(
            BaseLinearTrainerConfig config,
            LossFunctionType lossFunctionType,
            float learningRate,
            float learningRateDecay,
            int numThreads)
    {
        super(config);
        this.lossFunctionType = lossFunctionType;
        this.learningRate = learningRate;
        this.learningRateDecay = learningRateDecay;
        this.numThreads = numThreads;
    }

    public static SgdTrainerConfig fromCommandLineArguments(CommandLineArguments args) {
        return new SgdTrainerConfig(BaseLinearTrainerConfig.fromCommandLineArguments(args), args.lossFunctionType,
                args.learningRate, args.learningRateDecay, args.numThreads);
    }

    @Override
    public OptimizationAlgorithmType getOptimizationType() {
        return OptimizationAlgorithmType.SGD;
    }
}
