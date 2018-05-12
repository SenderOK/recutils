package ru.recutils.io;

import java.io.Serializable;

import ru.recutils.exceptions.DatasetLineParsingException;

public interface StringToFeaturesHolderConverter<FeaturesHolderT> extends Serializable {
    FeaturesHolderT convert(String s) throws DatasetLineParsingException;
}
