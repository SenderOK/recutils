package ru.recutils.common;

import java.util.Random;

public class MathUtils {
    public static float dotProduct(float first[], float second[], int size) {
        float result = 0;
        for (int i = 0; i < size; ++i) {
            result += first[i] * second[i];
        }
        return result;
    }

    public static void inplaceAddWithScale(float[] result, float[] add, float scale, int size) {
        for (int i = 0; i < size; ++i) {
            result[i] += add[i] * scale;
        }
    }

    public static void inplaceScale(float[] result, float scale) {
        for (int i = 0; i < result.length; ++i) {
            result[i] *= scale;
        }
    }

    public static float[] getRandomGaussianArray(Random randomGen, float stddev, int size) {
        float[] embedding =  new float[size];
        for (int i = 0; i < embedding.length; ++i) {
            embedding[i] = (float)randomGen.nextGaussian() * stddev;
        }
        return embedding;
    }

    public static float[] add(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    public static float l2normSquared(float[] a) {
        return dotProduct(a, a, a.length);
    }
}
