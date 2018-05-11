package ru.recutils.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.Spliterators;

import ru.recutils.exceptions.DatasetLineParsingException;

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
            return Files.lines(Paths.get(filename)).parallel().map(this::parseLine).iterator();
        } catch (IOException ex) {
            ex.printStackTrace();
            return Collections.emptyIterator();
        }
    }

    @Override
    public Spliterator<FeaturesHolderT> spliterator() {
        try {
            return Files.lines(Paths.get(filename)).parallel().map(this::parseLine).spliterator();
        } catch (IOException ex) {
            ex.printStackTrace();
            return Spliterators.emptySpliterator();
        }
    }

    private FeaturesHolderT parseLine(String line) {
        try {
            return converter.convert(line);
        } catch (DatasetLineParsingException ex) {
            System.err.println(ex.getMessage());
            return null;
        }
    }
}
