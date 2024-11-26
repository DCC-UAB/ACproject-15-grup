import os
import cv2
import pickle
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

def train_test(dataset, train_size=0.6, test_size=0.2, val_size=0.2):
    train, val, test = {}, {}, {}
    
    for class_name, images in dataset.items():
        # Dividir el conjunt entre dades d'entrenament i (validació + test)
        train_images, temp_images = train_test_split(images, train_size=train_size, random_state=42)

        # Dividir el conjunt (validació + test) entre dades de validació i test
        val_images, test_images = train_test_split(temp_images, test_size=test_size / (val_size + test_size), random_state=42)
        
        train[class_name] = train_images
        val[class_name] = val_images
        test[class_name] = test_images
    
    return train, val, test




# data = load_dataset('data/Cervical_Cancer')
# train_set, val_set, test_set = train_test(data)
# print(test_set)

    