package ru.recutils.common;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Stream;

import ru.recutils.exceptions.DatasetLineParsingException;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.lossfuncs.LossFunction;

public class Utils {

    public static <T extends ObservationHolder> List<Float> predict(
            String dataPath,
            StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter,
            LinearModelWeights modelWeights,
            LossFunction lossFunction)
    {
        List<Float> result = new ArrayList<>();

        DoubleAdder lossSum = new DoubleAdder();
        AtomicInteger objectCount = new AtomicInteger(0);
        try (Stream<String> dataStream = Files.lines(Paths.get(dataPath)).parallel()) {
            dataStream.forEach(line -> {
                T observation;
                try {
                    observation = stringToFeaturesHolderConverter.convert(line);
                } catch (DatasetLineParsingException ex) {
                    System.err.println(ex.getMessage());
                    return;
                }

                float prediction = modelWeights.apply(observation);
                float loss = lossFunction.value(prediction, observation.getLabel());
                lossSum.add(loss);
                objectCount.incrementAndGet();
            });
            dataStream.close();
        } catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }

        if (objectCount.get() > 0) {
            System.out.println("Average loss on " + objectCount + " objects is "
                    + lossSum.sum() / objectCount.get());
        } else {
            System.out.println("No objects processed");
        }

        return result;
    }

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
}
