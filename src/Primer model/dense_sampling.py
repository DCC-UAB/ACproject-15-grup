import cv2 as cv
from load_dataset import *
import numpy as np
import pickle


def dense_sampling(sift, imatges, labels, pases, amplada_punt):
    """
    Aplica el mètode de dense sampling a les imatges passades com a paràmetre, creant una màscara de punts uniforme 
    per a totes les imatges.

    :param sift: objecte SIFT
    :param imatges: np.array amb les imatges (en escala de grisos)
    :param labels: np.array amb les etiquetes de les imatges
    :param pases: int amb la distància entre punts de la màscara
    :param amplada_punt: int que indica la mida del keypoint
    :return: Descriptors imatges i diccionari amb els descriptors per categoria
    """
    
    height, width = imatges.shape[0], imatges.shape[1]
    keypoints = []
    for i in range(0, height, pases):
        for j in range(0, width, pases):
            keypoints.append(cv.KeyPoint(j, i, amplada_punt))

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