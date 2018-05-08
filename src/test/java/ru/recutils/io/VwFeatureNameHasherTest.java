package ru.recutils.io;

import junit.framework.Assert;
import junit.framework.TestCase;
import ru.recutils.exceptions.InvalidHashBitSizeException;

public class VwFeatureNameHasherTest extends TestCase {
    public void testHashes() throws InvalidHashBitSizeException {
        String featureName1 = "featureName";
        String featureName2 = "featureName";
        for (int i = 1; i < 31; ++i) {
            VwFeatureNameHasher hasher = VwFeatureNameHasher.getHasher(i);
            Assert.assertEquals(hasher.getHash(featureName1), hasher.getHash(featureName2));
            System.out.println(i + ":" + hasher.getHash(featureName1));
        }
    }
}
