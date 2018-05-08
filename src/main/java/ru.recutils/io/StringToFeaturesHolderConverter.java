package ru.recutils.io;

import ru.recutils.exceptions.DatasetLineParsingException;

public interface StringToFeaturesHolderConverter<FeaturesHolderT> {
    FeaturesHolderT convert(String s) throws DatasetLineParsingException;
}
