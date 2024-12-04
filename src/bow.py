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

def bag_of_words_histogram(features, n_clusters=1024):
    # Create the bag of words
    #dubte sobre si bow ha de ser una matriu o ha de ser una llista de matrius unidimensionals
    try: 
        with open("data/bow.pkl", "rb") as f:
            return pickle.load(f)
    except:
        kmeans = train_visual_words(features, n_clusters)
        bow = np.zeros((len(features), kmeans.n_clusters))
        for label, image_feature in features.items():
            for descriptor in image_feature:
                print(descriptor)
                pred = kmeans.predict(descriptor)
                for i in pred:
                    bow[label][i] += 1
        return bow

def main():
    data, labels = load_dataset('data/Cervical_Cancer')
    labels_encoded = encode_labels(labels)
    # try:
    #     with open('data/features.pkl', 'rb') as f:
    #         pass
    #         # categories = pickle.load(f)
    # except:
    #     # vector, categories = extract_sift_features(data, labels, 128, None)
    #     with open('data/features.pkl', 'wb') as f:
    #         # pickle.dump(categories, f)
    #         pass
    #vector, categories = extract_sift_features(data, labels_encoded, 128, None)

    features = extract_sift_features(data, labels_encoded, 128, None)
    bow = bag_of_words_histogram(features)
    print(bow)



if __name__ == '__main__':
    # main()
    pass
    