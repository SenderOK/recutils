package ru.recutils.trainers.fm;

import ru.recutils.common.ObservationHolder;
import ru.recutils.trainers.BaseLinearTrainerConfig;

public class FmAlsTrainer {
    public static <T extends ObservationHolder> void train(
            Iterable<T> dataset,
            FmModelWeights fmModelWeights,
            FmModelConfig modelConfig,
            BaseLinearTrainerConfig trainerConfig)
    {
    }
}
