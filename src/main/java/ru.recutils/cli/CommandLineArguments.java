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
            description = "model type: REG, FM, FFM (case insensitive)")
    public ModelType modelType = ModelType.REGRESSION;

    @Parameter(names={"-opt", "--optimizer"}, converter = OptimizationAlgorithmTypeConverter.class,
            description = "SGD, ALS (only for FM and FFM models) (case insensitive)")
    public OptimizationAlgorithmType optimizationAlgorithmType = OptimizationAlgorithmType.SGD;

    @Parameter(names = {"-l2"}, validateWith = PositiveDouble.class,
            description = "l2 regularizer for feature weights")
    public double featureWeightsRegularizer = 0.1;

    @Parameter(names = {"-l2e"}, validateWith = PositiveDouble.class,
            description = "l2 regularizer for feature embeddings (only for FM and FFM models)")
    public double embeddingsRegularizer = 0.1;

    @Parameter(names = {"-r", "--learning-rate"}, validateWith = PositiveDouble.class,
            description = "learning rate (for SGD training)")
    public double learningRate = 0.01;

    @Parameter(names = {"-iter"}, validateWith = PositiveInteger.class,
            description = "dataset passes for SGD, number of iterations for ALS training")
    public int numIter = 100;

    @Parameter(names = {"-b", "--batch-size"}, validateWith = PositiveInteger.class,
            description = "number of training objects in batch (for SGD training)")
    public int batchSize = 1;

    @Parameter(names={"-hb", "--hashing-bits"}, validateWith = PositiveInteger.class,
            description = "number of bits used for features hashing")
    public int hashingBits = 18;

    @Parameter(names={"-d", "--dimension"}, validateWith = PositiveInteger.class,
            description = "embeddings dimensionality in FM and FFM (for training)")
    public int dimension = 20;

    @Parameter(names={"-t", "--threads"}, validateWith = PositiveInteger.class,
            description = "number of threads (using Hogwild! for SGD training)")
    public int numThreads = 1;

    @Parameter(names={"-init", "--init-stddev"}, validateWith = PositiveDouble.class,
            description = "standard deviation for initial random embeddings")
    public double initStddev = 0.1;
}