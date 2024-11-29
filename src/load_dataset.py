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
            dataset, labels = pickle.load(f)
    except:
        dataset = []
        labels = []
        for root, dirs, files in os.walk(path):
            for dir_name in dirs[:3]:  # Only take the first 3 directories
                folder_path = os.path.join(root, dir_name)
                
                images = []
                for file in os.listdir(folder_path)[:200]:  # Only take the first 200 images
                    img_path = os.path.join(folder_path, file)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    if img is not None:
                        img_resized = img[::10, ::10]  # Take one pixel every 10 pixels
                        dataset.append(img_resized)
                        labels.append(dir_name)
        with open('data/dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
    return np.array(dataset), labels

def preprocess_images(image_paths, target_size=(32, 32)):
    images = []
    categories = {k:v for k, v in enumerate(image_paths.keys())}
    for category in image_paths.values():
        for path in category[:200]:
            try:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    img = img / 255.0
                    images.append(img)
            except Exception as e:
                print(f"Error processant {path}: {e}")
    return np.array(images), categories


def train_test(dataset, labels, test_size=0.2, val_size=0.2):
    X_train, X_val_test, y_train, y_val_test = train_test_split(dataset, labels, train_size=val_size+test_size, random_state=42)
    # print(len(X_train), len(y_train), len(X_val_test), len(y_val_test))
    X_test, X_val, y_test, y_val = train_test_split(X_val_test, y_val_test, test_size=val_size/(test_size + val_size), random_state=42)
    # ore = LabelEncoder(list(dataset.keys()))
    # y_train_encoded = ore.fit_transform(y_train)
    # y_test_encoded = ore.transform(y_test)
    # y_val_encoded = ore.transform(y_val)

    encoders = LabelEncoder()
    encoders.fit(list(set(y_train+y_test+y_val)))
    # Encoding the variable
    y_train_encoded = encoders.transform(y_train) #Per tranformar els str a numeros
    y_test_encoded = encoders.transform(y_test)
    y_val_encoded = encoders.transform(y_val)

    return X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded



#data, labels = load_dataset('data/Cervical_Cancer')
#print(len(data), len(labels))

#X_train, y_train, X_val, y_val, X_test, y_test = train_test(data, labels)
#for x, y in zip(X_train, y_train):
#    print("Imatge:", x, "Tipus de cancer:", y)