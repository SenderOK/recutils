package ru.recutils.common;

import ru.recutils.lossfuncs.BinaryLogisticLoss;
import ru.recutils.lossfuncs.LossFunction;
import ru.recutils.lossfuncs.SquaredLoss;

public enum LossFunctionType {
    MSE("mse"),
    LOGLOSS("logloss")
    ;

    private String name;

    LossFunctionType(String name) {
        this.name = name;
    }

    public String getName() {
        return this.name;
    }

    public static LossFunctionType fromString(String model) {
        for (LossFunctionType lossFunctionType : LossFunctionType.values()) {
            if (model.equalsIgnoreCase(lossFunctionType.name)) {
                return lossFunctionType;
            }
        }
        return null;
    }

    public LossFunction getLossFunction() {
        if (this == MSE) {
            return new SquaredLoss();
        } else if (this == LOGLOSS) {
            return new BinaryLogisticLoss();
        }
        return null;
    }
}
