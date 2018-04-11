package ru.recutils.trainers;

import java.io.Serializable;
import java.util.List;

import ru.recutils.common.BaseLinearTrainerConfig;
import ru.recutils.common.HashedLinearModel;
import ru.recutils.exceptions.ModelNotTrainedException;
import ru.recutils.common.ModelType;
import ru.recutils.common.ObservationHolder;
import ru.recutils.io.FeatureNameHasher;

public class FmModel<T extends ObservationHolder> implements HashedLinearModel<T>, Serializable {
    private final FeatureNameHasher featureNameHasher;
    private final BaseLinearTrainerConfig config;
    private final FmModelWeights modelWeights;
    private boolean wasTrained;

    public FmModel(FeatureNameHasher featureNameHasher, BaseLinearTrainerConfig config) {
        this.featureNameHasher = featureNameHasher;
        this.config = config;
        this.modelWeights = new FmModelWeights();
        this.wasTrained = false;
    }

    @Override
    public void fit(Iterable<T> dataset) {

        wasTrained = true;
    }

    @Override
    public List<Double> predict(Iterable<T> dataset) throws ModelNotTrainedException {
        if (!wasTrained) {
            throw new ModelNotTrainedException();
        }
        return null;
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
