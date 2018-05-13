package ru.recutils.trainers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Stream;

import ru.recutils.common.LinearModelWeights;
import ru.recutils.common.ObservationHolder;
import ru.recutils.exceptions.DatasetLineParsingException;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.lossfuncs.LossFunction;

public abstract class SgdTrainer<T extends ObservationHolder, ModelWeightsT extends LinearModelWeights, ModelConfigT> {
    protected final SgdTrainerConfig trainerConfig;
    protected final Random randomGen;
    protected final ForkJoinPool forkJoinPool;
    protected final LossFunction lossFunction;
    protected final StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter;

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

    public boolean train(String dataPath, ModelWeightsT modelWeights, ModelConfigT modelConfig) throws IOException {
        for (int i = 0; i < trainerConfig.numIter; ++i) {
            System.out.println("training epoch #" + i);

            DoubleAdder lossSum = new DoubleAdder();
            AtomicInteger objectCount = new AtomicInteger(0);
            DoubleAdder holdoutLossSum = new DoubleAdder();
            AtomicInteger holdoutObjectCount = new AtomicInteger(0);
            try (Stream<String> dataStream = Files.lines(Paths.get(dataPath)).parallel()) {
                forkJoinPool.submit(() -> dataStream.forEach(line -> {
                    T observation;
                    try {
                        observation = stringToFeaturesHolderConverter.convert(line);
                    } catch (DatasetLineParsingException ex) {
                        System.err.println(ex.getMessage());
                        return;
                    }

                    if (trainerConfig.useHoldout && Math.abs(line.hashCode()) % 10 == 0) {
                        float loss = lossFunction.value(modelWeights.apply(observation), observation.getLabel());
                        holdoutLossSum.add(loss);
                        holdoutObjectCount.incrementAndGet();
                    } else {
                        float loss = updateModelWeightsAndReturnLoss(observation, modelWeights, modelConfig);
                        lossSum.add(loss);
                        objectCount.incrementAndGet();
                    }
                })).get();
            } catch (ExecutionException|InterruptedException ex) {
                ex.printStackTrace();
                System.exit(1);
                return false;
            }

            if (objectCount.get() > 0) {
                System.out.print("Average train loss on " + objectCount.get() + " objects is "
                        + lossSum.sum() / objectCount.get());
                if (trainerConfig.useHoldout) {
                    System.out.print(" Average holdout loss on " + holdoutObjectCount.get() + " objects is "
                            + holdoutLossSum.sum() / holdoutObjectCount.get());
                }
                System.out.println("");
            } else {
                System.out.println("No objects processed");
            }
        }
        return true;
    }

    public abstract float updateModelWeightsAndReturnLoss(T observation, ModelWeightsT modelWeights, ModelConfigT modelConfig);
}
