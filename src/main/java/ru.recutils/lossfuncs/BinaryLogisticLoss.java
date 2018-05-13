package ru.recutils.lossfuncs;

public class BinaryLogisticLoss implements LossFunction {

    @Override
    public float value(float prediction, float gt) {
        float margin = prediction * gt;
        // logsumexp trick
        if (margin < 0) {
            return (float)Math.log1p(Math.exp(-margin));
        } else {
            return -margin + (float)Math.log1p(Math.exp(margin));
        }
    }

    @Override
    public float derivative(float prediction, float gt) {
        return -gt / (1 + (float)Math.exp(gt * prediction));
    }
}
