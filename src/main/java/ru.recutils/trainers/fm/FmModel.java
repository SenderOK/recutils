package ru.recutils.trainers.fm;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;

import ru.recutils.common.HashedLinearModel;
import ru.recutils.common.LossFunctionType;
import ru.recutils.common.OptimizationAlgorithmType;
import ru.recutils.common.Utils;
import ru.recutils.exceptions.ModelNotTrainedException;
import ru.recutils.common.ModelType;
import ru.recutils.common.ObservationHolder;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.lossfuncs.LossFunction;
import ru.recutils.trainers.BaseLinearTrainerConfig;
import ru.recutils.trainers.SgdTrainerConfig;

public class FmModel<T extends ObservationHolder> implements HashedLinearModel, Serializable {
    private final StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter;
    private final FmModelConfig fmModelConfig;
    private final BaseLinearTrainerConfig trainerConfig;
    private final FmModelWeights modelWeights;
    private boolean wasTrained;

    public FmModel(
            StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter,
            FmModelConfig fmModelConfig,
            BaseLinearTrainerConfig trainerConfig)
    {
        this.stringToFeaturesHolderConverter = stringToFeaturesHolderConverter;
        this.fmModelConfig = fmModelConfig;
        this.trainerConfig = trainerConfig;
        this.modelWeights = new FmModelWeights(fmModelConfig.dimension);
        this.wasTrained = false;
    }

    @Override
    public void fit(String dataPath) throws IOException {
        if (trainerConfig.getOptimizationType() == OptimizationAlgorithmType.SGD) {
            this.wasTrained = new FmSgdTrainer<T>((SgdTrainerConfig) trainerConfig, stringToFeaturesHolderConverter)
                    .train(dataPath, modelWeights, fmModelConfig);
        } else {
            this.wasTrained = new FmAlsTrainer<T>(trainerConfig, stringToFeaturesHolderConverter)
                    .train(dataPath, modelWeights, fmModelConfig);
        }
    }

    @Override
    public List<Float> predict(String dataPath) throws ModelNotTrainedException, IOException {
        if (!wasTrained) {
            throw new ModelNotTrainedException();
        }

        LossFunction lossFunction = (trainerConfig.getOptimizationType() == OptimizationAlgorithmType.SGD)
                ? ((SgdTrainerConfig) trainerConfig).lossFunctionType.getLossFunction()
                : LossFunctionType.MSE.getLossFunction();

        return Utils.predict(dataPath, stringToFeaturesHolderConverter, modelWeights, lossFunction);
    }

    @Override
    public ModelType getModelType() {
        return ModelType.FM;
    }
}
