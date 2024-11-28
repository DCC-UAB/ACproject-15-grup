import os
import cv2
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(path):
    try:
        with open('data/dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    except:

        dataset = {}
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                
                images = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                    if cv2.imread(os.path.join(folder_path, file)) is not None]
                dataset[dir_name] = images
        with open('data/dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
    return dataset

def preprocess_images(image_paths, target_size=(128, 128)):
    images = []
    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                images.append(img)
        except Exception as e:
            print(f"Error processant {path}: {e}")
    return np.array(images)


def train_test(dataset, train_size=0.6, test_size=0.2, val_size=0.2):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    for class_name, images in dataset.items():
        # Dividir el conjunt entre dades d'entrenament i (validació + test)
        train_images, temp_images = train_test_split(images, train_size=train_size, random_state=42)

        # Dividir el conjunt (validació + test) entre dades de validació i test
        val_images, test_images = train_test_split(temp_images, test_size=test_size / (val_size + test_size), random_state=42)
        
        X_train.extend(train_images)
        y_train.extend([class_name] * len(train_images))
        
        X_val.extend(val_images)
        y_val.extend([class_name] * len(val_images))
        
        X_test.extend(test_images)
        y_test.extend([class_name] * len(test_images))
    
    return X_train, y_train, X_val, y_val, X_test, y_test




# data = load_dataset('data/Cervical_Cancer')
# X_train, y_train, X_val, y_val, X_test, y_test = train_test(data)
# for x, y in zip(X_train, y_train):
#     print("Imatge:", x, "Tipus de cancer:", y)

    