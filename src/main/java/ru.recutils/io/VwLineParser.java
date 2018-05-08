package ru.recutils.io;

import java.util.HashMap;

import ru.recutils.exceptions.DatasetLineParsingException;

public class VwLineParser implements StringToFeaturesHolderConverter<SimpleObservationHolder> {
    // parsing format from https://github.com/JohnLangford/vowpal_wabbit/wiki/input-format
    // currently without tag, base and namespaces

    private final FeatureNameHasher featureNameHasher;

    public VwLineParser(FeatureNameHasher featureNameHasher) {
        this.featureNameHasher = featureNameHasher;
    }

    @Override
    public SimpleObservationHolder convert(String line) throws DatasetLineParsingException, NumberFormatException {
        String[] features = line.split("\\s+");
        if (features.length <= 2) {
            throw new DatasetLineParsingException("Invalid line format");
        }

        double label = Double.parseDouble(features[0]);

        int currIndex = 1;
        double importance = 1.0;
        if (!features[1].equals("|")) {
            importance = Double.parseDouble(features[1]);
            currIndex = 2;
        }

        if (!features[currIndex].equals("|")) {
            throw new DatasetLineParsingException("'|' symbol expected before the features");
        }

        HashMap<Integer, Double> featureWeights = new HashMap<>(features.length - currIndex - 1);
        for (int i = currIndex + 1; i < features.length; ++i) {
            String[] featureParts = features[i].split(":");

            if (!(1 <= featureParts.length && featureParts.length <= 2)) {
                throw new DatasetLineParsingException("Invalid features format : '" + features[i] + "'");
            }

            double featureValue = (featureParts.length == 2) ? Double.parseDouble(featureParts[1]) : 1.0;
            featureWeights.merge(featureNameHasher.getHash(featureParts[0]), featureValue, (a, b) -> a + b);
        }

        return new SimpleObservationHolder(featureWeights, label, importance);
    }
}
