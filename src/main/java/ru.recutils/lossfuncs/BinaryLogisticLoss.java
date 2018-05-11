package ru.recutils.lossfuncs;

public class BinaryLogisticLoss implements LossFunction {

    @Override
    public float value(float prediction, float gt) {
        float exp = -prediction * gt;
        // logsumexp trick
        if (exp < 0) {
            return (float)Math.log1p(Math.exp(exp));
        } else {
            return exp + (float)Math.log1p(Math.exp(-exp));
        }
    }

    @Override
    public float derivative(float prediction, float gt) {
        return gt / (1 + (float)Math.exp(gt * prediction));
    }
}
