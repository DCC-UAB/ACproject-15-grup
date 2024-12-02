import numpy as np
from sift import extract_sift_features
from load_dataset import load_dataset
import pickle

def bag_of_words(features, vocab_size=1024):
    # Create the bag of words
    bow = np.zeros((len(features), vocab_size))
    for i, f in enumerate(features):
        for descriptor in f:
            for j in descriptor:
                if int(j) < vocab_size:
                    bow[i][int(j)] += 1
    return bow

# # Example usage
# data, labels = load_dataset('data/Cervical_Cancer')
# try:
#     with open('data/features.pkl', 'rb') as f:
#         features = pickle.load(f)
# except:
#     features = extract_sift_features(data)
#     with open('data/features.pkl', 'wb') as f:
#         pickle.dump(features, f)

# # print(features)
# bow = bag_of_words(features)
# print(bow)
