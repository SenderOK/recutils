package ru.recutils.cli;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.ParameterException;

import ru.recutils.common.OptimizationAlgorithmType;

public class OptimizationAlgorithmTypeConverter implements IStringConverter<OptimizationAlgorithmType> {
    @Override
    public OptimizationAlgorithmType convert(String value) throws ParameterException {
        OptimizationAlgorithmType optimizationAlgorithmType = OptimizationAlgorithmType.fromString(value);
        if (optimizationAlgorithmType == null) {
            throw new ParameterException("invalid optimization algorithm");
        }
        return optimizationAlgorithmType;
    }
}
