package ru.recutils.common;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class HashedLinearModelLoader {
    public static HashedLinearModel load(String path) {
        try (
                FileInputStream fileInputStream = new FileInputStream(path);
                ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
        ) {
            return (HashedLinearModel) objectInputStream.readObject();
        } catch (Exception ex) {
            ex.printStackTrace();
            System.exit(1);
        }
        return null;
    }
}
