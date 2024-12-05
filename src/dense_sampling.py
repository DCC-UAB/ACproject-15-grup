import cv2 as cv
from load_dataset import *
from sift import extract_sift_features
import numpy as np
import pickle


def dense_sampling(imatges, labels, pases, amplada_punt, nfeatures ):
    
    height, width = imatges.shape[0], imatges.shape[1]
    mascara = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, pases):
        for j in range(0, width, pases):
            mascara[i:i+amplada_punt, j:j+amplada_punt] = 1
    categories = extract_sift_features(imatges, labels, nfeatures, mascara)
    return categories

def main():
    data, labels = load_dataset('data/Cervical_Cancer')

    try:
        with open('data/dense_sampling_features.pkl', 'rb') as f:
            features = pickle.load(f)
    except:
        vector, features = dense_sampling(data, labels, 15, 5, 128) #descriptors = [keypoints, descriptors]???
        #recordatori: quan fem proves per diferents escales, crear pickle per cada escala
        with open('data/dense_sampling_features.pkl', 'wb') as f:
            pickle.dump(features, f)
    print(features)

if __name__ == '__main__':
    main()
    pass