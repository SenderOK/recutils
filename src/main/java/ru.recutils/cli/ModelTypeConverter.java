package ru.recutils.cli;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.ParameterException;

import ru.recutils.common.ModelType;

public class ModelTypeConverter implements IStringConverter<ModelType> {
    @Override
    public ModelType convert(String value) throws ParameterException {
        ModelType modelType = ModelType.fromString(value);
        if (modelType == null) {
            throw new ParameterException("invalid model type");
        }
        return modelType;
    }
}
