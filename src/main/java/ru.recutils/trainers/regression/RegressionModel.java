package ru.recutils.trainers.regression;

import java.util.ArrayList;
import java.util.List;

import ru.recutils.common.HashedLinearModel;
import ru.recutils.exceptions.ModelNotTrainedException;
import ru.recutils.common.ModelType;
import ru.recutils.common.ObservationHolder;
import ru.recutils.io.FeatureNameHasher;
import ru.recutils.lossfuncs.LossFunction;
import ru.recutils.trainers.SgdTrainerConfig;

public class RegressionModel<T extends ObservationHolder> implements HashedLinearModel<T> {
    private final FeatureNameHasher featureNameHasher;
    private final RegressionModelConfig regressionModelConfig;
    private final SgdTrainerConfig sgdTrainerConfig;
    private final RegressionModelWeights modelWeights;
    private boolean wasTrained;

    public RegressionModel(
            FeatureNameHasher featureNameHasher,
            RegressionModelConfig regressionModelConfig,
            SgdTrainerConfig sgdTrainerConfig)
    {
        this.featureNameHasher = featureNameHasher;
        this.regressionModelConfig = regressionModelConfig;
        this.sgdTrainerConfig = sgdTrainerConfig;
        this.modelWeights = new RegressionModelWeights();
        this.wasTrained = false;
    }

    @Override
    public void fit(Iterable<T> dataset) {
        RegressionSgdTrainer.train(dataset, modelWeights, regressionModelConfig, sgdTrainerConfig);
        this.wasTrained = true;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.REGRESSION;
    }

    @Override
    public FeatureNameHasher getFeatureNameHasher() {
        return featureNameHasher;
    }

    @Override
    public List<Double> predict(Iterable<T> dataset) throws ModelNotTrainedException {
        if (!wasTrained) {
            throw new ModelNotTrainedException();
        }
        List<Double> result = new ArrayList<>();
        double lossSum = 0;
        int objectCount = 0;
        LossFunction lossFunction = sgdTrainerConfig.lossFunctionType.getLossFunction();
        for (ObservationHolder observation : dataset) {
            double prediction = modelWeights.apply(observation);
            lossSum += lossFunction.value(prediction, observation.getLabel());
            ++objectCount;
            result.add(prediction);
        }
        System.out.println("Average loss on " + objectCount + " objects is " + lossSum / objectCount);
        return result;
    }
}
