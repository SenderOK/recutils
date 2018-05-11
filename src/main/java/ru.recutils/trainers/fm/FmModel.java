package ru.recutils.trainers.fm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import ru.recutils.common.HashedLinearModel;
import ru.recutils.common.LossFunctionType;
import ru.recutils.common.OptimizationAlgorithmType;
import ru.recutils.exceptions.ModelNotTrainedException;
import ru.recutils.common.ModelType;
import ru.recutils.common.ObservationHolder;
import ru.recutils.io.FeatureNameHasher;
import ru.recutils.lossfuncs.LossFunction;
import ru.recutils.trainers.BaseLinearTrainerConfig;
import ru.recutils.trainers.SgdTrainerConfig;

public class FmModel<T extends ObservationHolder> implements HashedLinearModel<T>, Serializable {
    private final FeatureNameHasher featureNameHasher;
    private final FmModelConfig fmModelConfig;
    private final BaseLinearTrainerConfig trainerConfig;
    private final FmModelWeights modelWeights;
    private boolean wasTrained;

    public FmModel(
            FeatureNameHasher featureNameHasher,
            FmModelConfig fmModelConfig,
            BaseLinearTrainerConfig trainerConfig)
    {
        this.featureNameHasher = featureNameHasher;
        this.fmModelConfig = fmModelConfig;
        this.trainerConfig = trainerConfig;
        this.modelWeights = new FmModelWeights(fmModelConfig.dimension);
        this.wasTrained = false;
    }

    @Override
    public void fit(Iterable<T> dataset) {
        if (trainerConfig.getOptimizationType() == OptimizationAlgorithmType.SGD) {
            FmSgdTrainer.train(dataset, modelWeights, fmModelConfig, (SgdTrainerConfig) trainerConfig);
        } else {
            FmAlsTrainer.train(dataset, modelWeights, fmModelConfig, trainerConfig);
        }
        this.wasTrained = true;
    }

    @Override
    public List<Float> predict(Iterable<T> dataset) throws ModelNotTrainedException {
        if (!wasTrained) {
            throw new ModelNotTrainedException();
        }
        LossFunction lossFunction = (trainerConfig.getOptimizationType() == OptimizationAlgorithmType.SGD)
                ? ((SgdTrainerConfig) trainerConfig).lossFunctionType.getLossFunction()
                : LossFunctionType.MSE.getLossFunction();

        List<Float> result = new ArrayList<>();
        float lossSum = 0;
        int objectCount = 0;
        for (ObservationHolder observation : dataset) {
            float prediction = modelWeights.apply(observation);
            lossSum += lossFunction.value(prediction, observation.getLabel());
            ++objectCount;
            result.add(prediction);
        }
        System.out.println("Average loss on " + objectCount + " objects is " + lossSum / objectCount);
        return result;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.FM;
    }

    @Override
    public FeatureNameHasher getFeatureNameHasher() {
        return featureNameHasher;
    }
}
