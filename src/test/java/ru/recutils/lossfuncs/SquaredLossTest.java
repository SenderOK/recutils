package ru.recutils.lossfuncs;

import junit.framework.Assert;
import junit.framework.TestCase;

public class SquaredLossTest extends TestCase {
    public void testValueAndDerivative() {
        LossFunction loss = new SquaredLoss();
        Assert.assertEquals(25, loss.value(5, 10), 1e-6);
        Assert.assertEquals(-10, loss.derivative(5, 10), 1e-6);
    }
}
