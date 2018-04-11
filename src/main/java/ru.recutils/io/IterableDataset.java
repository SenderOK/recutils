package ru.recutils.io;

import java.io.IOException;
import java.util.Collections;
import java.util.Iterator;

public class IterableDataset<FeaturesHolderT> implements Iterable<FeaturesHolderT> {
    private final String filename;
    private final StringToFeaturesHolderConverter<FeaturesHolderT> converter;

    public IterableDataset(String filename, StringToFeaturesHolderConverter<FeaturesHolderT> converter) {
        this.filename = filename;
        this.converter = converter;
    }

    @Override
    public Iterator<FeaturesHolderT> iterator() {
        try {
            return new DatasetIterator<>(filename, converter);
        } catch (IOException ex) {
            ex.printStackTrace();
            return Collections.emptyIterator();
        }
    }
}
