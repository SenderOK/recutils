package ru.recutils.lossfuncs;

public class BinaryLogisticLoss implements LossFunction {

    @Override
    public double value(double prediction, double gt) {
        double exp = -prediction * gt;
        // logsumexp trick
        if (exp < 0) {
            return Math.log1p(Math.exp(exp));
        } else {
            return (exp + Math.log1p(Math.exp(-exp)));
        }
    }

    @Override
    public double derivative(double prediction, double gt) {
        return gt / (1 + Math.exp(gt * prediction));
    }
}
