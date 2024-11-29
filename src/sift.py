import cv2
from load_dataset import *
import pickle

def sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def extract_sift_features(images):
    features = []
    for image in images:
        keypoints, descriptors = sift_features(image)
        features.append(descriptors)
    return features

def main():
    data, labels = load_dataset('data/Cervical_Cancer')
    try:
        with open('data/features.pkl', 'rb') as f:
            features = pickle.load(f)
    except:
        print("Extracció de característiques SIFT...")
        features = extract_sift_features(data)
        print(f"Característiques SIFT extretes: {len(features)}")
        with open('data/features.pkl', 'wb') as f:
            pickle.dump(features, f)


if __name__ == '__main__':
    main()

