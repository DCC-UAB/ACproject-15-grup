import numpy as np
from sift import extract_sift_features
from load_dataset import *
from sklearn.cluster import KMeans
import pickle

def train_visual_words(vector_features, n_clusters=1024):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # desc = list(features.values())
    # values_array = np.concatenate(desc, axis=0)

    kmeans.fit(np.array(vector_features))
    return kmeans

def bag_of_words_histogram(features, kmeans): #mateix kmeans per train i test
    bow = []
    # kmeans = train_visual_words(vector, n_clusters)
    labels = []
    for label, image_feature in features.items():
        for descriptor in image_feature: #descriptors de la imatge
            hist_label = np.zeros(shape = kmeans.n_clusters) 
            pred = kmeans.predict(descriptor).astype(int) #predim a quin cluster correspon cada descriptor
            for i in pred:
                hist_label[i] += 1 #comptem quants descriptors tenim a cada cluster
            bow.append(hist_label)
            labels.append(label)
    bow = np.array(bow)
    labels = np.array(labels)
    return bow, labels

def main():
    data, labels = load_dataset('data/Cervical_Cancer')
    labels_encoded = encode_labels(labels)

    try:
        with open("data/bow_sift.pkl", "rb") as f:
            bow = pickle.load(f)
            print(bow)
    except:
        vectors, features = extract_sift_features(data, labels_encoded, 128, None)
        bow = bag_of_words_histogram(vectors, features)
        print(bow)



if __name__ == '__main__':
    # main()
    pass
    