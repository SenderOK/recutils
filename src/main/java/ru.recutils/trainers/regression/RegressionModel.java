package ru.recutils.trainers.regression;

import java.io.IOException;
import java.util.List;

import ru.recutils.common.HashedLinearModel;
import ru.recutils.common.Utils;
import ru.recutils.exceptions.ModelNotTrainedException;
import ru.recutils.common.ModelType;
import ru.recutils.common.ObservationHolder;
import ru.recutils.io.StringToFeaturesHolderConverter;
import ru.recutils.lossfuncs.LossFunction;
import ru.recutils.trainers.SgdTrainerConfig;

public class RegressionModel<T extends ObservationHolder> implements HashedLinearModel {
    private final StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter;
    private final RegressionModelConfig regressionModelConfig;
    private final SgdTrainerConfig sgdTrainerConfig;
    private final RegressionModelWeights modelWeights;
    private boolean wasTrained;

    public RegressionModel(
            StringToFeaturesHolderConverter<T> stringToFeaturesHolderConverter,
            RegressionModelConfig regressionModelConfig,
            SgdTrainerConfig sgdTrainerConfig)
    {
        this.stringToFeaturesHolderConverter = stringToFeaturesHolderConverter;
        this.regressionModelConfig = regressionModelConfig;
        this.sgdTrainerConfig = sgdTrainerConfig;
        this.modelWeights = new RegressionModelWeights();
        this.wasTrained = false;
    }

    @Override
    public void fit(String dataPath) throws IOException {
        this.wasTrained = new RegressionSgdTrainer<T>(sgdTrainerConfig, stringToFeaturesHolderConverter)
                .train(dataPath, modelWeights, regressionModelConfig);
    }

    @Override
    public ModelType getModelType() {
        return ModelType.REGRESSION;
    }

    @Override
    public List<Float> predict(String dataPath) throws ModelNotTrainedException, IOException {
        if (!wasTrained) {
            throw new ModelNotTrainedException();
        }
        LossFunction lossFunction = sgdTrainerConfig.lossFunctionType.getLossFunction();
        return Utils.predict(dataPath, stringToFeaturesHolderConverter, modelWeights, lossFunction);
    }
}
