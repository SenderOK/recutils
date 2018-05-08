package ru.recutils.trainers;

import java.util.Map;

import com.sun.tools.javac.util.Assert;

import ru.recutils.common.ObservationHolder;
import ru.recutils.lossfuncs.LossFunction;

public class RegressionSgdTrainer {
    public static <T extends ObservationHolder> void train(
            Iterable<T> dataset,
            RegressionModelWeights regressionModelWeights,
            SgdTrainerConfig config)
    {
        LossFunction lossFunction = config.lossFunctionType.getLossFunction();
        Assert.check(lossFunction != null);

        for (int i = 0; i < config.numIter; ++i) {
            System.out.println("training epoch #" + i);

            double lossSum = 0;
            int objectCount = 0;
            for (T observation : dataset) {
                double prediction = regressionModelWeights.apply(observation);
                double label = observation.getLabel();
                lossSum += lossFunction.value(prediction, label);
                ++objectCount;

                regressionModelWeights.bias -= config.learningRate * lossFunction.derivative(prediction, label);

                for (Map.Entry<Integer, Double> entry : observation.getFeatures().entrySet()) {
                    int featureHash = entry.getKey();
                    double featureValue = entry.getValue();
                    double gradient = lossFunction.derivative(prediction, label) * featureValue +
                            config.featureWeightsRegularizer *
                                    regressionModelWeights.featureWeights.getOrDefault(featureHash, 0.0);
                    regressionModelWeights.featureWeights.merge(featureHash, -config.learningRate * gradient,
                            (a, b) -> a + b);
                }
            }

            System.out.println("Average loss on " + objectCount + " objects is " + lossSum / objectCount);
        }
    }
}
