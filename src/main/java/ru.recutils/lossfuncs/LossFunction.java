package ru.recutils.lossfuncs;

public interface LossFunction {
    double value(double prediction, double gt);

    double derivative(double prediction, double gt);
}
