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

        double learningRate = config.learningRate;
        for (int i = 0; i < config.numIter; ++i) {
            for (T observation : dataset) {
                double prediction = regressionModelWeights.apply(observation);
                double label = observation.getLabel();
                regressionModelWeights.bias -= learningRate * lossFunction.derivative(prediction, label);

                for (Map.Entry<Integer, Double> entry : observation.getFeatures().entrySet()) {
                    double gradient = lossFunction.derivative(prediction, label) * entry.getValue();
                    regressionModelWeights.featureWeights.merge(entry.getKey(), -gradient, (a, b) -> a + b);
                }
            }
        }
    }
}
