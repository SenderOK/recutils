package ru.recutils.cli;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.ParameterException;

import ru.recutils.common.LossFunctionType;

public class LossFunctionTypeConverter implements IStringConverter<LossFunctionType> {
    @Override
    public LossFunctionType convert(String value) throws ParameterException {
        LossFunctionType lossFunctionType = LossFunctionType.fromString(value);
        if (lossFunctionType == null) {
            throw new ParameterException("invalid loss function");
        }
        return lossFunctionType;
    }
}
