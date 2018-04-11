package ru.recutils.io;

public interface StringToFeaturesHolderConverter<FeaturesHolderT> {
    FeaturesHolderT convert(String s);
}
