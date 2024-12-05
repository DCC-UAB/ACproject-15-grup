import cv2
from load_dataset import *
import pickle
import numpy as np
from collections import defaultdict



def extract_sift_features(images, labels, n, mask=None):
    sift = cv2.SIFT_create(nfeatures=n)
    vector = []
    categories = defaultdict(list) #Diccionari de llistes a on guardem la categoria de cada feature
    for image, label in zip(images, labels):
        _ , descriptors = sift.detectAndCompute(image, mask)
        vector.extend(descriptors)
        categories[label].append(descriptors)

    vector = np.array(vector)
    return vector, categories


def main():
    n = 128 #Això és l'escala de les característiques SIFT
    data, labels = load_dataset('data/Cervical_Cancer')
    try:
        with open('data/features.pkl', 'rb') as f:
            categories = pickle.load(f)
    except:
        print("Extracció de característiques SIFT...")
        vector, categories = extract_sift_features(data, labels, n, None)
        # print(f"Característiques SIFT extretes: {len(features)}")
        print(categories)
        #recordatori: quan fem proves per diferents escales, crear pickle per cada escala

        with open('data/features.pkl', 'wb') as f:
            pickle.dump(categories, f)


if __name__ == '__main__':
    main()
    pass

