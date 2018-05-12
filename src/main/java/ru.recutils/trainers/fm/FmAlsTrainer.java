package ru.recutils.trainers.fm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import javafx.util.Pair;
import ru.recutils.common.ObservationHolder;
import ru.recutils.common.Utils;
import ru.recutils.exceptions.DatasetLineParsingException;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.trainers.BaseLinearTrainerConfig;

public class FmAlsTrainer<T extends ObservationHolder> {
    public final BaseLinearTrainerConfig trainerConfig;
    public final Random randomGen;
    public final StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter;

    public FmAlsTrainer(
            BaseLinearTrainerConfig trainerConfig,
            StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter)
    {
        this.trainerConfig = trainerConfig;
        this.randomGen = new Random(trainerConfig.seed);
        this.stringToFeaturesHolderConverter = stringToFeaturesHolderConverter;
    }

    public boolean train(String dataPath, FmModelWeights modelWeights, FmModelConfig modelConfig) throws IOException {
        Map<Integer, List<Pair<Integer, Float>>> featureHashToObservations = new HashMap<>();
        List<Float> errors = new ArrayList<>();
        List<float[]> weightedEmbeddingsSums = new ArrayList<>();
        int embeddingSize = modelConfig.dimension;

        System.out.println("Reading the dataset");
        AtomicInteger i = new AtomicInteger();
        Files.lines(Paths.get(dataPath)).forEach(line -> {
            T observation;
            try {
                observation = stringToFeaturesHolderConverter.convert(line);
            } catch (DatasetLineParsingException ex) {
                System.err.println(ex.getMessage());
                return;
            }

            modelWeights.initializeZeroWeights(observation, randomGen, trainerConfig.initStddev);
            errors.add(modelWeights.apply(observation) - observation.getLabel());

            // precalculating weighted sum of embedding vectors
            float[] weightedEmbeddingsSum = new float[embeddingSize];
            for (Map.Entry<Integer, Float> entry : observation.getFeatures().entrySet()) {
                int featureHash = entry.getKey();
                float featureValue = entry.getValue();

                Utils.inplaceAddWithScale(
                        weightedEmbeddingsSum,
                        modelWeights.featureEmbeddings.get(featureHash),
                        featureValue,
                        embeddingSize
                );

                // updating inverted index of observations
                featureHashToObservations.computeIfAbsent(featureHash, v -> new ArrayList<>())
                        .add(new Pair<>(i.get(), featureValue));
            }
            weightedEmbeddingsSums.add(weightedEmbeddingsSum);
            i.incrementAndGet();
        });

        System.out.println("Starting ALS iterations, initial MSE is " + getMSE(errors) + " for "
                + i.get() + " objects");
        for (int iter = 0; iter < trainerConfig.numIter; ++iter) {
            System.out.println("training epoch #" + i);
            alsStep(modelWeights, modelConfig, featureHashToObservations, errors, weightedEmbeddingsSums);
            System.out.println("Loss: " + getMSE(errors));
        }

        return true;
    }

    private void alsStep(
            FmModelWeights modelWeights,
            FmModelConfig modelConfig,
            Map<Integer, List<Pair<Integer, Float>>> featureHashToObservations,
            List<Float> errors,
            List<float[]> weightedEmbeddingSums)
    {
        updateBias(modelWeights, errors);
        updateWeights(modelWeights, modelConfig, errors, featureHashToObservations);
        updateEmbeddings(modelWeights, modelConfig, errors, featureHashToObservations, weightedEmbeddingSums);
    }

    private void updateBias(FmModelWeights modelWeights, List<Float> errors) {
        float errorSum = 0.0f;
        int trainObservations = 0;
        for (int i = 0; i < errors.size(); ++i) {
            if (trainerConfig.useHoldout && i % 10 == 0) {
                continue;
            }
            errorSum += errors.get(i);
            ++trainObservations;
        }
        float oldBias = modelWeights.regressionModelWeights.bias;
        float newBias = oldBias - errorSum / trainObservations;

        // updating bias and errors
        modelWeights.regressionModelWeights.bias = newBias;
        for (int i = 0; i < errors.size(); ++i) {
            errors.set(i, errors.get(i) + (newBias - oldBias));
        }
    }

    private void updateWeights(
            FmModelWeights modelWeights,
            FmModelConfig modelConfig,
            List<Float> errors,
            Map<Integer, List<Pair<Integer, Float>>> featureHashToObservations)
    {
        for (Map.Entry<Integer, List<Pair<Integer, Float>>> entry : featureHashToObservations.entrySet()) {
            Integer featureHash = entry.getKey();
            float oldWeight = modelWeights.regressionModelWeights.featureWeights.get(featureHash);

            float numerator = 0.0f;
            float denominator = 0.0f;
            int trainObservations = 0;
            for (Pair<Integer, Float> objectIndexFeatureValuePair : entry.getValue()) {
                int objectIdx = objectIndexFeatureValuePair.getKey();
                if (trainerConfig.useHoldout && objectIdx % 10 == 0) {
                    continue;
                }
                float featureValue = objectIndexFeatureValuePair.getValue();

                numerator += (errors.get(objectIdx) - oldWeight * featureValue) * featureValue;
                denominator += featureValue * featureValue;
                ++trainObservations;
            }

            float newWeight = - numerator / (denominator + trainObservations * modelConfig.featureWeightsRegularizer);

            // updating weight and errors
            modelWeights.regressionModelWeights.featureWeights.put(featureHash, newWeight);
            for (Pair<Integer, Float> objectIndexFeatureValuePair : entry.getValue()) {
                int objectIdx = objectIndexFeatureValuePair.getKey();
                float featureValue = objectIndexFeatureValuePair.getValue();
                errors.set(objectIdx, errors.get(objectIdx) + (newWeight - oldWeight) * featureValue);
            }
        }
    }

    private void updateEmbeddings(
            FmModelWeights modelWeights,
            FmModelConfig modelConfig,
            List<Float> errors,
            Map<Integer, List<Pair<Integer, Float>>> featureHashToObservations,
            List<float[]> weightedEmbeddingSums)
    {
        for (int dim = 0; dim < modelConfig.dimension; ++dim) {
            for (Map.Entry<Integer, List<Pair<Integer, Float>>> entry : featureHashToObservations.entrySet()) {
                Integer featureHash = entry.getKey();
                float oldWeight = modelWeights.featureEmbeddings.get(featureHash)[dim];

                float numerator = 0.0f;
                float denominator = 0.0f;
                int trainObservations = 0;
                for (Pair<Integer, Float> objectIndexFeatureValuePair : entry.getValue()) {
                    int objectIdx = objectIndexFeatureValuePair.getKey();
                    if (trainerConfig.useHoldout && objectIdx % 10 == 0) {
                        continue;
                    }
                    float featureValue = objectIndexFeatureValuePair.getValue();
                    float h = (weightedEmbeddingSums.get(objectIdx)[dim] - oldWeight * featureValue) * oldWeight;

                    numerator += (errors.get(objectIdx) - oldWeight * h) * h;
                    denominator += h * h;
                    ++trainObservations;
                }

                float newWeight = - numerator / (denominator + trainObservations * modelConfig.embeddingsRegularizer);

                // updating weight, errors and weighted embedding
                modelWeights.featureEmbeddings.get(featureHash)[dim] = newWeight;
                for (Pair<Integer, Float> objectIndexFeatureValuePair : entry.getValue()) {
                    int objectIdx = objectIndexFeatureValuePair.getKey();
                    float featureValue = objectIndexFeatureValuePair.getValue();
                    float h = (weightedEmbeddingSums.get(objectIdx)[dim] - oldWeight * featureValue) * oldWeight;

                    errors.set(objectIdx, errors.get(objectIdx) + (newWeight - oldWeight) * h);
                    weightedEmbeddingSums.get(objectIdx)[dim] += (newWeight - oldWeight) * featureValue;
                }
            }
        }
    }

    private static float getMSE(List<Float> errors) {
        float result = 0.0f;
        for (float error : errors) {
            result += error * error;
        }
        return result / errors.size();
    }
}
