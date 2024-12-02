import cv2 as cv
from load_dataset import *
def dense_sampling(imatge, pases, amplada_punt):
    height, width = imatge.shape[0], imatge.shape[1]
    seleccionar_punts = []
    for i in range(0, height, pases):
        for j in range(0, width, pases):
            seleccionar_punts.append(cv.KeyPoint(j, i, amplada_punt))
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.compute(imatge, seleccionar_punts)
    return keypoints, descriptors

def main():
    data, labels = load_dataset('data/Cervical_Cancer')

    try:
        with open('data/dense_sampling_features.pkl', 'rb') as f:
            keypoints, descriptors = pickle.load(f)
    except:
        keypoints, descriptors = dense_sampling(data, 10, 10)
        with open('data/dense_sampling_features.pkl', 'wb') as f:
            pickle.dump(descriptors, f)
    print(descriptors)

if __name__ == '__main__':
    # main()
    pass