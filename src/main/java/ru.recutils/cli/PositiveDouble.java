package ru.recutils.cli;

import com.beust.jcommander.IParameterValidator;
import com.beust.jcommander.ParameterException;

public class PositiveDouble implements IParameterValidator {
    @Override
    public void validate(String name, String value) throws ParameterException {
        double x = Double.parseDouble(value);
        if (Double.compare(x, 0) <= 0) {
            throw new ParameterException("Parameter " + name + " should be positive (found " + value +")");
        }
    }
}
