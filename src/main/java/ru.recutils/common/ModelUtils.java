package ru.recutils.common;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import ru.recutils.exceptions.DatasetLineParsingException;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.lossfuncs.LossFunction;

public class ModelUtils {
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
            result = dataStream.map(line -> {
                T observation;
                try {
                    observation = stringToFeaturesHolderConverter.convert(line);
                } catch (DatasetLineParsingException ex) {
                    System.err.println(ex.getMessage());
                    return 0.0f;
                }

                float prediction = modelWeights.apply(observation);
                float loss = lossFunction.value(prediction, observation.getLabel());
                lossSum.add(loss);
                objectCount.incrementAndGet();
                return prediction;
            }).collect(Collectors.toList());
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
}
