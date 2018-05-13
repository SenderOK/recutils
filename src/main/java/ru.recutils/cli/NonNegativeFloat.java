package ru.recutils.cli;

import com.beust.jcommander.IParameterValidator;
import com.beust.jcommander.ParameterException;

public class NonNegativeFloat implements IParameterValidator {
    @Override
    public void validate(String name, String value) throws ParameterException {
        float x = Float.parseFloat(value);
        if (Float.compare(x, 0) < 0) {
            throw new ParameterException("Parameter " + name + " should be non-negative (found " + value +")");
        }
    }
}
