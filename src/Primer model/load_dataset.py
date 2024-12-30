import os
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_dataset(path, num_directoris, num_dades):
    """
    Es carrega el dataset de les imatges i es transformen en escala de grisos.
    Es crea un pickel amb les imatges i les etiquetes.

    :param path: path on es troben les imatges
    :return: np.array amb les imatges i una llista amb les etiquetes
    """
    try:
        with open('data/dataset2.pkl', 'rb') as f:
            dataset, labels = pickle.load(f)  # Assegurem que es carreguen tant dataset com labels
    except:
        dataset = []
        labels = []
        for root, dirs, _ in os.walk(path):
            for dir_name in dirs[:num_directoris]:
                folder_path = os.path.join(root, dir_name)
                for file in os.listdir(folder_path)[:num_dades]:
                    img_path = os.path.join(folder_path, file)
                    img = cv2.imread(img_path)

                    if img is not None:
                        img_resized = img
                        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                        dataset.append(img_gray)
                        labels.append(dir_name)

        # Guarda tant dataset com labels al pickle
        with open('data/dataset2.pkl', 'wb') as f:
            pickle.dump((dataset, labels), f)
    return np.array(dataset), labels


def encode_labels(labels):
    """
    Codifica les etiquetes en valors numèrics.

    :param labels: llista amb les etiquetes
    :return: np.array amb les etiquetes codificades
    """
    encoders = LabelEncoder()
    encoders.fit(list(set(labels)))
    labels_encoded = encoders.transform(labels)
    return labels_encoded


def train_test(dataset, labels, test_size=0.2, val_size=0):
    """
    Retorna els conjunts d'entrenament, validació i test.
    En el cas que no es vulgui validació, es retorna None. Valor per defecte de validació és 0, i test 0.2.

    :param dataset: np.array amb les imatges
    :param labels: np.array amb les etiquetes
    :param test_size: float amb la mida del conjunt de test
    :param val_size: float amb la mida del conjunt de validació
    :return: np.array amb les imatges d'entrenament, validació i test i les etiquetes d'entrenament, validació i test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=(test_size + val_size), random_state=42)
    if val_size != 0:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (test_size + val_size)), random_state=42)
        return X_train, y_train, X_test, y_test, X_val, y_val 
    
    return X_train, y_train, X_temp, y_temp, None, None