package ru.recutils.lossfuncs;

import junit.framework.Assert;
import junit.framework.TestCase;

public class BinaryLogisticLossTest extends TestCase {
    public void testValueAndDerivative() {
        LossFunction loss = new BinaryLogisticLoss();
        Assert.assertEquals(Math.log(2.0f), loss.value(0, 1), 1e-6);
        Assert.assertEquals(-3, loss.derivative(0, 6), 1e-6);
    }
}
