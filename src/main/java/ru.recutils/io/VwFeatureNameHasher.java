package ru.recutils.io;

import com.sangupta.murmur.Murmur3;
import com.sun.tools.javac.util.Assert;

public class VwFeatureNameHasher implements FeatureNameHasher {
    private static final long SEED = 1543154315431543L;

    private final int bitMask;

    private VwFeatureNameHasher(int bitMask) {
        this.bitMask = bitMask;
    }

    public static VwFeatureNameHasher getHasher(int featuresHashBitSize) throws IndexOutOfBoundsException {
        Assert.check(featuresHashBitSize > 0 && featuresHashBitSize < 31, "hash bit size too big");
        int bitMask = (1 << featuresHashBitSize) - 1;
        return new VwFeatureNameHasher(bitMask);
    }

    public int getHash(String featureName) {
        byte[] data = featureName.getBytes();
        long hash = Murmur3.hash_x86_32(data, data.length, SEED);
        return (int)(hash & (long)bitMask);
    }
}
