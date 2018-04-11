package ru.recutils.lossfuncs;

public class SquaredLoss implements LossFunction {
    @Override
    public double value(double prediction, double gt) {
        return (prediction - gt) * (prediction - gt);
    }

    @Override
    public double derivative(double prediction, double gt) {
        return 2 * (prediction - gt);
    }
}
