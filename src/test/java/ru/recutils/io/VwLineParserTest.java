package ru.recutils.io;

import java.util.Map;

import junit.framework.Assert;
import junit.framework.TestCase;
import ru.recutils.exceptions.DatasetLineParsingException;
import ru.recutils.exceptions.InvalidHashBitSizeException;

public class VwLineParserTest extends TestCase {
    public void testOkLineParsing1() throws InvalidHashBitSizeException, DatasetLineParsingException {
        VwFeatureNameHasher hasher = VwFeatureNameHasher.getHasher(20);
        VwLineParser parser = new VwLineParser(hasher);
        SimpleObservationHolder holder = parser.convert("-2\t20.0  | f1:15 f2:43");
        Map<Integer, Double> features = holder.getFeatures();
        Assert.assertEquals(features.get(hasher.getHash("f1")), 15.0);
        Assert.assertEquals(features.get(hasher.getHash("f2")), 43.0);
        Assert.assertEquals(holder.getLabel(), -2.0);
        Assert.assertEquals(holder.getImportance(), 20.0);
        System.out.println(holder.getFeatures());
    }

    public void testOkLineParsing2() throws InvalidHashBitSizeException, DatasetLineParsingException {
        VwFeatureNameHasher hasher = VwFeatureNameHasher.getHasher(20);
        VwLineParser parser = new VwLineParser(hasher);
        SimpleObservationHolder holder = parser.convert("1  |  f1");
        Map<Integer, Double> features = holder.getFeatures();
        Assert.assertEquals(features.get(hasher.getHash("f1")), 1.0);
        Assert.assertEquals(holder.getLabel(), 1.0);
        Assert.assertEquals(holder.getImportance(), 1.0);
        System.out.println(holder.getFeatures());
    }

    public void testInvalidLineFormatParsing1() throws InvalidHashBitSizeException {
        VwLineParser parser = new VwLineParser(VwFeatureNameHasher.getHasher(20));

        String errorMessage = "";
        try {
            parser.convert("1.0 | ");
        } catch (DatasetLineParsingException error) {
            errorMessage = error.getMessage();
        }
        Assert.assertEquals("Invalid line format", errorMessage);
    }

    public void testInvalidLineFormatParsing2() throws InvalidHashBitSizeException {
        VwLineParser parser = new VwLineParser(VwFeatureNameHasher.getHasher(20));
        String errorMessage = "";
        try {
            parser.convert("1.0 3.0 + f1");
        } catch (DatasetLineParsingException error) {
            errorMessage = error.getMessage();
        }
        Assert.assertEquals("'|' symbol expected before the features", errorMessage);
    }

    public void testInvalidLineFormatParsing3() throws InvalidHashBitSizeException {
        VwLineParser parser = new VwLineParser(VwFeatureNameHasher.getHasher(20));

        String errorMessage = "";
        try {
            parser.convert("1.0 | ");
        } catch (DatasetLineParsingException error) {
            errorMessage = error.getMessage();
        }
        Assert.assertEquals("Invalid line format", errorMessage);
    }
}
