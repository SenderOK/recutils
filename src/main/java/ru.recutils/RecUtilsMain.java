package ru.recutils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import com.beust.jcommander.JCommander;

import ru.recutils.cli.CommandLineArguments;
import ru.recutils.common.BaseLinearTrainerConfig;
import ru.recutils.common.HashedLinearModel;
import ru.recutils.common.HashedLinearModelLoader;
import ru.recutils.common.LossFunctionType;
import ru.recutils.common.ModelType;
import ru.recutils.common.OptimizationAlgorithmType;
import ru.recutils.exceptions.InvalidHashBitSizeException;
import ru.recutils.exceptions.ModelNotTrainedException;
import ru.recutils.trainers.FmModel;
import ru.recutils.io.FeatureNameHasher;
import ru.recutils.io.IterableDataset;
import ru.recutils.io.SimpleObservationHolder;
import ru.recutils.io.VwFeatureNameHasher;
import ru.recutils.io.VwLineParser;
import ru.recutils.trainers.RegressionModel;
import ru.recutils.trainers.AlsTrainerConfig;
import ru.recutils.trainers.SgdTrainerConfig;

public class RecUtilsMain {
    public static void main(String[] args) throws InvalidHashBitSizeException {
        CommandLineArguments commandLineArguments = new CommandLineArguments();
        JCommander jCommander = new JCommander(commandLineArguments);
        jCommander.setCaseSensitiveOptions(false);
        jCommander.parse(args);

        if (commandLineArguments.help) {
            jCommander.usage();
            return;
        }

        if (commandLineArguments.trainPath == null && commandLineArguments.testPath == null) {
            System.err.println("Neither training set nor test set specified, exiting");
            return;
        }

        if (commandLineArguments.testPath != null && commandLineArguments.resultPath == null) {
            System.err.println("Path to the result file for predictions should be specified, exiting");
            return;
        }

        if (commandLineArguments.trainPath != null) {
            doTrain(commandLineArguments);
        }

        if (commandLineArguments.testPath != null) {
            doTest(commandLineArguments);
        }
    }

    private static void doTrain(CommandLineArguments args) throws InvalidHashBitSizeException {
        if (args.modelType == ModelType.REGRESSION && args.optimizationAlgorithmType != OptimizationAlgorithmType.SGD) {
            System.out.println("for regression task only sgd is available, will use sgd");
        }

        if (args.optimizationAlgorithmType == OptimizationAlgorithmType.ALS
                && args.lossFunctionType != LossFunctionType.MSE)
        {
            System.out.println("for als only mse function is available, will use mse");
        }

        FeatureNameHasher featureNameHasher = VwFeatureNameHasher.getHasher(args.hashingBits);
        HashedLinearModel<SimpleObservationHolder> hashedLinearModel;
        if (args.modelType == ModelType.REGRESSION) {
            SgdTrainerConfig sgdTrainerConfig = SgdTrainerConfig.fromCommandLineArguments(args);
            hashedLinearModel = new RegressionModel<SimpleObservationHolder>(featureNameHasher, sgdTrainerConfig);
        } else if (args.modelType == ModelType.FM) {
            BaseLinearTrainerConfig linearTrainerConfig;
            if (args.optimizationAlgorithmType == OptimizationAlgorithmType.SGD) {
                linearTrainerConfig = SgdTrainerConfig.fromCommandLineArguments(args);
            } else { // ALS
                linearTrainerConfig = AlsTrainerConfig.fromCommandLineArguments(args);
            }
            hashedLinearModel = new FmModel<SimpleObservationHolder>(featureNameHasher, linearTrainerConfig);
        } else {
            System.err.println("FFM is not implemented yet");
            return;
        }

        IterableDataset<SimpleObservationHolder> dataset = new IterableDataset<>(args.trainPath,
                new VwLineParser(featureNameHasher));
        hashedLinearModel.fit(dataset);

        System.out.println("Fit model on file " + args.trainPath);

        hashedLinearModel.dump(args.modelPath);

        System.out.println("Dumped the model to " + args.modelPath);
    }

    private static void doTest(CommandLineArguments args) {
        HashedLinearModel hashedLinearModel = HashedLinearModelLoader.load(args.modelPath);
        if (hashedLinearModel == null) {
            System.err.println("Failed to load the model");
            return;
        }

        FeatureNameHasher featureNameHasher = hashedLinearModel.getFeatureNameHasher();
        IterableDataset<SimpleObservationHolder> dataset = new IterableDataset<>(args.testPath,
                new VwLineParser(featureNameHasher));
        List<Double> predictions;
        try {
            predictions = hashedLinearModel.predict(dataset);
        } catch (ModelNotTrainedException exception) {
            System.err.println("Could not make predictions, model was not trained");
            return;
        }

        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(args.resultPath))) {
            for (double prediction : predictions) {
                bufferedWriter.write(Double.toString(prediction) + "\n");
            }
        } catch (IOException e) {
            System.err.println("Failed to write predictions");
        }

        System.out.println("Successfully wrote the predictions to " + args.resultPath);
    }
}
