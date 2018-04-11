package ru.recutils.common;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;

import ru.recutils.exceptions.ModelNotTrainedException;
import ru.recutils.io.FeatureNameHasher;

public interface HashedLinearModel<ObservationT> extends Serializable {
    void fit(Iterable<ObservationT> dataset);

    List<Double> predict(Iterable<ObservationT> dataset) throws ModelNotTrainedException;

    ModelType getModelType();

    FeatureNameHasher getFeatureNameHasher();

    default void dump(String modelPath) {
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(modelPath);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        ) {
            objectOutputStream.writeObject(this);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
