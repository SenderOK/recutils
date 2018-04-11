package ru.recutils.trainers;

import java.io.Serializable;

import ru.recutils.cli.CommandLineArguments;
import ru.recutils.common.BaseLinearTrainerConfig;

public class AlsTrainerConfig extends BaseLinearTrainerConfig implements Serializable {
    public final double initStddev;
    public final double embeddingsRegularizer;
    public final int dimension;

    public AlsTrainerConfig(
            BaseLinearTrainerConfig config,
            double initStddev,
            double embeddingsRegularizer,
            int dimension)
    {
        super(config);
        this.initStddev = initStddev;
        this.embeddingsRegularizer = embeddingsRegularizer;
        this.dimension = dimension;
    }

    public static AlsTrainerConfig fromCommandLineArguments(CommandLineArguments args) {
        return new AlsTrainerConfig(BaseLinearTrainerConfig.fromCommandLineArguments(args), args.initStddev,
                args.embeddingsRegularizer, args.dimension);
    }
}
