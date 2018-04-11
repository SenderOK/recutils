package ru.recutils.common;

public enum ModelType {
    REGRESSION("reg"),
    FM("fm"),
    FFM("ffm")
    ;

    private String name;

    ModelType(String name) {
        this.name = name;
    }

    public String getName() {
        return this.name;
    }

    public static ModelType fromString(String model) {
        for (ModelType modelType : ModelType.values()) {
            if (model.equalsIgnoreCase(modelType.name)) {
                return modelType;
            }
        }
        return null;
    }
}
