from load_dataset import *
import pickle
import numpy as np
from collections import OrderedDict


def extract_sift_features(sift, images, labels): #nfeatures eliminat
    """
    Extreu les característiques SIFT de les imatges passades com a paràmetre.
    
    :param sift: objecte SIFT
    :param images: np.array amb les imatges (en escala de grisos)
    :param labels: np.array amb les etiquetes de les imatges
    :return: np.array amb els descriptors de les imatges i diccionari amb els descriptors
    """
    vector = []
    categories = OrderedDict() #Diccionari de llistes a on guardem la categoria de cada feature
    for image, label in zip(images, labels):
        if label not in categories.keys():
            categories[label] = []
        _ , descriptors = sift.detectAndCompute(image, mask=None)
        if descriptors is not None:
            vector.extend(descriptors)
            categories[label].append(descriptors)

    vector = np.asarray(vector) #np.array sol canvia ordre? provar
    return vector.astype(np.float32), categories


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

