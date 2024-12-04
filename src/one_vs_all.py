from bow import bag_of_words_histogram
from load_dataset import *
from sift import extract_sift_features

def one_vs_all(bow, labels_encoded, model):
    results = {}
    return results

def main():
    data, labels = load_dataset('data/Cervical_Cancer')
    labels_encoded = encode_labels(labels)
    try:
        with open('data/features.pkl', 'rb') as f:
            features = pickle.load(f)
    except:
        features = extract_sift_features(data, labels, 128, None)
        with open('data/features.pkl', 'wb') as f:
            pickle.dump(features, f)
    
    bow = bag_of_words_histogram(features)


    
