package ru.recutils.trainers;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Stream;

import ru.recutils.common.ObservationHolder;
import ru.recutils.exceptions.DatasetLineParsingException;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.lossfuncs.LossFunction;

public abstract class SgdTrainer<T extends ObservationHolder, ModelWeightsT, ModelConfigT> {
    public final SgdTrainerConfig trainerConfig;
    public final Random randomGen;
    public final ForkJoinPool forkJoinPool;
    public final LossFunction lossFunction;
    public final StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter;

    public SgdTrainer(
            SgdTrainerConfig trainerConfig,
            StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter)
    {
        this.trainerConfig = trainerConfig;
        this.randomGen = new Random(trainerConfig.seed);
        this.forkJoinPool = new ForkJoinPool(trainerConfig.numThreads);
        this.lossFunction = trainerConfig.lossFunctionType.getLossFunction();
        this.stringToFeaturesHolderConverter = stringToFeaturesHolderConverter;
    }

    public boolean train(String dataPath, ModelWeightsT modelWeights, ModelConfigT modelConfig) {
        for (int i = 0; i < trainerConfig.numIter; ++i) {
            System.out.println("training epoch #" + i);

            DoubleAdder lossSum = new DoubleAdder();
            AtomicInteger objectCount = new AtomicInteger(0);
            try (Stream<String> dataStream = Files.lines(Paths.get(dataPath)).parallel()) {
                forkJoinPool.submit(() -> dataStream.forEach(line -> {
                    T observation;
                    try {
                        observation = stringToFeaturesHolderConverter.convert(line);
                    } catch (DatasetLineParsingException ex) {
                        System.err.println(ex.getMessage());
                        return;
                    }

                    float loss = updateModelWeightsAndReturnLoss(observation, modelWeights, modelConfig);
                    lossSum.add(loss);
                    objectCount.incrementAndGet();
                })).get();
            } catch (Exception ex) {
                ex.printStackTrace();
                return false;
            }

            if (objectCount.get() > 0) {
                System.out.println("Average loss on " + objectCount + " objects is "
                        + lossSum.sum() / objectCount.get());
            } else {
                System.out.println("No objects processed");
            }
        }
        return true;
    }

    public abstract float updateModelWeightsAndReturnLoss(T observation, ModelWeightsT modelWeights, ModelConfigT modelConfig);
}
