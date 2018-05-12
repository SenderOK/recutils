package ru.recutils.cli;

import com.beust.jcommander.Parameter;

import ru.recutils.common.LossFunctionType;
import ru.recutils.common.ModelType;
import ru.recutils.common.OptimizationAlgorithmType;

public class CommandLineArguments {
    @Parameter(names = {"-h", "--help"}, help = true)
    public boolean help;

    @Parameter(names={"-m", "--model"}, required = true,
            description = "path to the model file (saving after training, using for testing)")
    public String modelPath;

    @Parameter(names={"-f", "--fit"}, description = "path to the file with training set")
    public String trainPath;

    @Parameter(names={"-p", "--predict"}, description = "path to the file with testing set")
    public String testPath;

    @Parameter(names={"-o", "--output"}, description = "path to the output file with predictions")
    public String resultPath;

    @Parameter(names={"-l", "--loss"}, converter = LossFunctionTypeConverter.class,
            description = "loss function for training: MSE, LOGLOSS for LOGLOSS, all labels must be +1 or -1")
    public LossFunctionType lossFunctionType = LossFunctionType.MSE;

    @Parameter(names={"-mt", "--model-type"}, converter = ModelTypeConverter.class,
            description = "model type: REG, FM (case insensitive)")
    public ModelType modelType = ModelType.REGRESSION;

    @Parameter(names={"-opt", "--optimizer"}, converter = OptimizationAlgorithmTypeConverter.class,
            description = "SGD, ALS (only for FM model) (case insensitive)")
    public OptimizationAlgorithmType optimizationAlgorithmType = OptimizationAlgorithmType.SGD;

    @Parameter(names = {"-l2"}, validateWith = PositiveFloat.class,
            description = "l2 regularizer for feature weights")
    public float featureWeightsRegularizer = 0.1f;

    @Parameter(names = {"-l2e"}, validateWith = PositiveFloat.class,
            description = "l2 regularizer for feature embeddings (only for FM model)")
    public float embeddingsRegularizer = 0.1f;

    @Parameter(names = {"-r", "--learning-rate"}, validateWith = PositiveFloat.class,
            description = "learning rate (for SGD training)")
    public float learningRate = 0.01f;

    @Parameter(names = {"-iter"}, validateWith = PositiveInteger.class,
            description = "dataset passes for SGD, number of iterations for ALS training")
    public int numIter = 100;

    @Parameter(names={"-hb", "--hashing-bits"}, validateWith = PositiveInteger.class,
            description = "number of bits used for features hashing")
    public int hashingBits = 18;

    @Parameter(names={"-d", "--dimension"}, validateWith = PositiveInteger.class,
            description = "embeddings dimensionality in FM (for training)")
    public int dimension = 20;

    @Parameter(names={"-t", "--threads"}, validateWith = PositiveInteger.class,
            description = "number of threads (using Hogwild! for SGD training)")
    public int numThreads = 1;

    @Parameter(names={"-init", "--init-stddev"}, validateWith = PositiveFloat.class,
            description = "standard deviation for initial weights")
    public float initStddev = 0.1f;

    @Parameter(names={"-s", "--seed"}, description = "random seed for training")
    public long seed = 42;

    @Parameter(names={"-h", "--holdout"}, description = "10% of training is used only for validation")
    public boolean useHoldout = false;
}