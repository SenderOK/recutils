package ru.recutils.lossfuncs;

public class SquaredLoss implements LossFunction {
    @Override
    public float value(float prediction, float gt) {
        return (prediction - gt) * (prediction - gt);
    }

    @Override
    public float derivative(float prediction, float gt) {
        return 2 * (prediction - gt);
    }
}
