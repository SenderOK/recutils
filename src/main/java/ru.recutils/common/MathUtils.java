package ru.recutils.common;

import java.util.Random;

public class MathUtils {
    public static double dotProduct(double first[], double second[], int size) {
        double result = 0;
        for (int i = 0; i < size; ++i) {
            result += first[i] * second[i];
        }
        return result;
    }

    public static void inplaceAddWithScale(double[] result, double[] add, double scale, int size) {
        for (int i = 0; i < size; ++i) {
            result[i] += add[i] * scale;
        }
    }

    public static void inplaceScale(double[] result, double scale) {
        for (int i = 0; i < result.length; ++i) {
            result[i] *= scale;
        }
    }

    public static double[] getRandomGaussianArray(Random randomGen, double stddev, int size) {
        double[] embedding =  new double[size];
        for (int i = 0; i < embedding.length; ++i) {
            embedding[i] = randomGen.nextGaussian() * stddev;
        }
        return embedding;
    }
}
