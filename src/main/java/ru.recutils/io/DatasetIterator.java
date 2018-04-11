package ru.recutils.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

public class DatasetIterator<FeaturesHolderT> implements Iterator<FeaturesHolderT> {
    private final StringToFeaturesHolderConverter<FeaturesHolderT> converter;
    private final BufferedReaderIterator bufferedReaderIterator;

    public DatasetIterator(
            String filename,
            StringToFeaturesHolderConverter<FeaturesHolderT> converter) throws IOException
    {
        this.converter = converter;
        this.bufferedReaderIterator = new BufferedReaderIterator(new BufferedReader(new FileReader(filename)));
    }

    @Override
    public boolean hasNext() {
        return bufferedReaderIterator.hasNext();
    }

    @Override
    public FeaturesHolderT next() {
        String s = this.bufferedReaderIterator.next();
        if (s == null) {
            return null;
        } else {
            return converter.convert(s);
        }
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("collection is read-only");
    }
}
