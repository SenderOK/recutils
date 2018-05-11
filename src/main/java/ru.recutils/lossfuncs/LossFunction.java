package ru.recutils.lossfuncs;

public interface LossFunction {
    float value(float prediction, float gt);

    float derivative(float prediction, float gt);
}
