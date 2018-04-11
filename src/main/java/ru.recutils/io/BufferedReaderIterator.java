package ru.recutils.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Iterator;

public class BufferedReaderIterator implements Iterator<String> {
    private final BufferedReader bufferedReader;

    public BufferedReaderIterator(final BufferedReader bufferedReader) {
        this.bufferedReader = bufferedReader;
    }

    @Override
    public boolean hasNext() {
        try {
            boolean result = this.bufferedReader.ready();
            if (!result) {
                close();
            }
            return result;
        } catch (IOException ex) {
            close();
            return false;
        }
    }

    @Override
    public String next() {
        try {
            return this.bufferedReader.readLine();
        } catch (final IOException e) {
            close();
            return null;
        }
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("collection is read-only");
    }

    private void close() {
        try {
            bufferedReader.close();
        } catch (IOException ex) {
            // do nothing
        }
    }

}