import cv2 as cv
from load_dataset import *
from sift import extract_sift_features
import numpy as np
import pickle


def dense_sampling(imatges, labels, pases, amplada_punt):
    
    height, width = imatges.shape[0], imatges.shape[1]
    keypoints = []
    for i in range(0, height, pases):
        for j in range(0, width, pases):
            keypoints.append(cv.KeyPoint(i, j, amplada_punt))

    sift = cv2.SIFT_create()
    vector = []
    categories = defaultdict(list) #Diccionari de llistes a on guardem la categoria de cada feature
    for image, label in zip(imatges, labels):
        _ , descriptors = sift.compute(image, keypoints)
        if descriptors is not None:
            vector.extend(descriptors)
            categories[label].append(descriptors)

    vector = np.array(vector)
    return vector, categories

def main():
    data, labels = load_dataset('data/Cervical_Cancer')

    try:
        with open('data/dense_sampling_features.pkl', 'rb') as f:
            vector, features = pickle.load(f)
    except:
        vector, features = dense_sampling(data, labels, 15, 5, 128)
        #recordatori: quan fem proves per diferents escales, crear pickle per cada escala
        with open('data/dense_sampling_features.pkl', 'wb') as f:
            pickle.dump((vector, features), f)

if __name__ == '__main__':
    main()
    pass