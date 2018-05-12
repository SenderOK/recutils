package ru.recutils.common;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;

import ru.recutils.exceptions.ModelNotTrainedException;

public interface HashedLinearModel extends Serializable {
    void fit(String dataPath);

    List<Float> predict(String dataPath) throws ModelNotTrainedException;

    ModelType getModelType();

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
