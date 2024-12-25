import os
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np
from collections import Counter


def load_dataset(path):
    try:
        with open('data/dataset.pkl', 'rb') as f:
            dataset, labels = pickle.load(f)  # Assegurem que es carreguen tant dataset com labels
    except:
        dataset = []
        labels = []
        for root, dirs, files in os.walk(path):
            for dir_name in dirs[:5]:  # Només agafa els primers 3 directoris
                folder_path = os.path.join(root, dir_name)
                
                for file in os.listdir(folder_path)[:300]:  # Només agafa les primeres 200 imatges
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


def train_test(dataset, labels, test_size=0.2, val_size=0):
    X_train, X_temp, y_train, y_temp = train_test_split(dataset, labels, test_size=(test_size + val_size), random_state=42)
    if val_size != 0:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (test_size + val_size)), random_state=42)
        return X_train, y_train, X_test, y_test, X_val, y_val 
    
    return X_train, y_train, X_temp, y_temp, None, None



# data, labels = load_dataset('data/Cervical_Cancer')
# print(len(data), len(labels))
# labels_encoded = encode_labels(labels)

# X_train, y_train, X_val, y_val, X_test, y_test = train_test(data, labels_encoded)
# def count_labels(labels):
#     return Counter(labels)

# train_counts = count_labels(y_train)
# val_counts = count_labels(y_val)
# test_counts = count_labels(y_test)

# print("Training set label counts:", train_counts)
# print("Validation set label counts:", val_counts)
# print("Test set label counts:", test_counts)