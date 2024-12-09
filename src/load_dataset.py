import os
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np


def load_dataset(path):
    try:
        with open('data/dataset.pkl', 'rb') as f:
            dataset, labels = pickle.load(f)  # Assegurem que es carreguen tant dataset com labels
    except:
        dataset = []
        labels = []
        for root, dirs, files in os.walk(path):
            for dir_name in dirs[:3]:  # Només agafa els primers 3 directoris
                folder_path = os.path.join(root, dir_name)
                
                for file in os.listdir(folder_path)[:200]:  # Només agafa les primeres 200 imatges
                    img_path = os.path.join(folder_path, file)
                    img = cv2.imread(img_path)

                    if img is not None:
                        img_resized = img
                        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                        dataset.append(img_gray)
                        labels.append(dir_name)
        # Guarda tant dataset com labels al pickle
        with open('data/dataset.pkl', 'wb') as f:
            pickle.dump((dataset, labels), f)
    return np.array(dataset), labels


def preprocess_images(image_paths, target_size=(32, 32)):
    images = []
    categories = {k:v for k, v in enumerate(image_paths.keys())}
    for category in image_paths.values():
        for path in category[:300]:
            try:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    img = img / 255.0
                    images.append(img)
            except Exception as e:
                print(f"Error processant {path}: {e}")
    return np.array(images), categories


def encode_labels(labels):
    encoders = LabelEncoder()
    encoders.fit(list(set(labels)))
    labels_encoded = encoders.transform(labels)
    return labels_encoded


def train_test(dataset, labels, test_size=0, val_size=0.2):
    X_train, X_val_test, y_train, y_val_test = train_test_split(dataset, labels, train_size=val_size+test_size, random_state=42)
    
    # X_test, X_val, y_test, y_val = train_test_split(X_val_test, y_val_test, test_size=test_size/(test_size + val_size), random_state=42)
    X_val, y_val = X_val_test, y_val_test

    return X_train, y_train, X_val, y_val, 0, 0



# data, labels = load_dataset('data/Cervical_Cancer')
# print(len(data), len(labels))
# labels_encoded = encode_labels(labels)

# X_train, y_train, X_val, y_val, X_test, y_test = train_test(data, labels_encoded)
# for x, y in zip(X_train, y_train):
#    print("Imatge:", x, "Tipus de cancer:", y)