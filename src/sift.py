import cv2
from load_dataset import *
import pickle


def sift_features(image, n, mask=None):
    sift = cv2.SIFT_create(nfeatures=n)
    keypoints, descriptors = sift.detectAndCompute(image, mask)
    return keypoints, descriptors

def extract_sift_features(images, n, mask=None):
    features = []
    for image in images:
        keypoints, descriptors = sift_features(image, n, mask)
        features.append(descriptors)
    return keypoints, features

def main():
    n = 128 #Això és l'escala de les característiques SIFT
    data, labels = load_dataset('data/Cervical_Cancer')
    try:
        with open('data/features.pkl', 'rb') as f:
            features = pickle.load(f)
    except:
        print("Extracció de característiques SIFT...")
        keypoints, features = extract_sift_features(data, n, None)
        print(f"Característiques SIFT extretes: {len(features)}")
        print(features)
        #recordatori: quan fem proves per diferents escales, crear pickle per cada escala

        with open('data/features.pkl', 'wb') as f:
            pickle.dump(features, f)


if __name__ == '__main__':
    # main()
    pass

