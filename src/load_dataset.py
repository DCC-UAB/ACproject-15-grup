import os
import cv2


def load_dataset(path):
    dataset = {}
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if cv2.imread(os.path.join(folder_path, file)) is not None]
            dataset[dir_name] = images
    return dataset

# print(load_dataset('data/Cervical_Cancer'))
    