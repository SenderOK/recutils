package ru.recutils.trainers;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;
import ru.recutils.common.BaseLinearTrainerConfig;
import ru.recutils.common.LossFunctionType;

public class SgdTrainerConfig extends BaseLinearTrainerConfig implements Serializable {
    public final LossFunctionType lossFunctionType;
    public final double learningRate;
    public final int batchSize;

    public SgdTrainerConfig(
            BaseLinearTrainerConfig baseLinearTrainerConfig,
            LossFunctionType lossFunctionType,
            double learningRate,
            int batchSize)
    {
        super(baseLinearTrainerConfig);
        this.lossFunctionType = lossFunctionType;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
    }

    public static SgdTrainerConfig fromCommandLineArguments(CommandLineArguments args) {
        return new SgdTrainerConfig(BaseLinearTrainerConfig.fromCommandLineArguments(args), args.lossFunctionType,
                args.learningRate, args.batchSize);
    }
}
