package ru.recutils.trainers;

import java.util.ArrayList;
import java.util.List;

import ru.recutils.common.HashedLinearModel;
import ru.recutils.exceptions.ModelNotTrainedException;
import ru.recutils.common.ModelType;
import ru.recutils.common.ObservationHolder;
import ru.recutils.io.FeatureNameHasher;

public class RegressionModel<T extends ObservationHolder> implements HashedLinearModel<T> {
    private final FeatureNameHasher featureNameHasher;
    private final SgdTrainerConfig sgdTrainerConfig;
    private final RegressionModelWeights regressionModelWeights;
    private boolean wasTrained;

    public RegressionModel(FeatureNameHasher featureNameHasher, SgdTrainerConfig sgdTrainerConfig) {
        this.featureNameHasher = featureNameHasher;
        this.sgdTrainerConfig = sgdTrainerConfig;
        this.regressionModelWeights = new RegressionModelWeights();
        this.wasTrained = false;
    }

    @Override
    public void fit(Iterable<T> dataset) {
        RegressionSgdTrainer.train(dataset, regressionModelWeights, sgdTrainerConfig);
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
        for (ObservationHolder observation : dataset) {
            result.add(regressionModelWeights.apply(observation));
        }
        return result;
    }
}
