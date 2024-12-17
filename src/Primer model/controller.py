from load_dataset import load_dataset
from load_dataset import encode_labels
from load_dataset import train_test
from sift import extract_sift_features
from bow import train_visual_words
from bow import bag_of_words_histogram
from train_test import train_logistic_regression
from train_test import train_svc
from sklearn import metrics
import cv2


def main():
    # sift = True
    print("Carregant i processant el dataset...")
    dataset_path = 'data/Cervical_Cancer'
    data, labels = load_dataset(dataset_path)
    labels_encoded = encode_labels(labels)
    X_train, y_train, X_val, y_val, X_test, y_test = train_test(data, labels_encoded)
    print("Extracció de característiques SIFT i creant histograma BoW...")
    
    # try:
    #     with open("data/bow_sift_train.pkl", 'rb') as f:
    #         bow_train = pickle.load(f)
  
    #     with open("data/bow_sift_test.pkl", 'rb') as f:
    #         bow_test = pickle.load(f)
    # except:
    print("Creant els BoW...")
    sift  = cv2.xfeatures2d.SIFT_create()

    vectors_train, train_features = extract_sift_features(sift, X_train, y_train)
    _, val_features = extract_sift_features(sift, X_val, y_val)

    kmeans = train_visual_words(vectors_train, 64)

    bow_train, labels_train = bag_of_words_histogram(train_features, kmeans)
    bow_val, labels_val = bag_of_words_histogram(val_features, kmeans)

    # vectors, features = dense_sampling(X_train, y_train, 10, 2)
    # kmeans = train_visual_words(vectors, 10)

    # bow_train, labels_train = bag_of_words_histogram(features, kmeans)

    # vectors, features = dense_sampling(X_test, y_test, 10, 2)
    # bow_test, labels_test = bag_of_words_histogram(features, kmeans)

        # with open("data/bow_sift_train.pkl", 'wb') as f:
        #     pickle.dump(bow_train, f)
        # with open("data/bow_sift_test.pkl", 'wb') as f:
        #     pickle.dump(bow_test, f)

    print("Entrenant el model...")
    # print(len(bow_train), len(y_train))
    model, best_params = train_logistic_regression(bow_train, labels_train)

    # model, best_params = train_svc(bow_train, labels_train)
    prediccio = model.predict(bow_val)
    print("confusion matrix: ",metrics.confusion_matrix(labels_val, prediccio))
    print("accuracy: ",metrics.accuracy_score(labels_val, prediccio))

    print("Params: ", best_params)
    
    print("Overfitting = ", model.score(bow_train, labels_train))
    print("Resultats test = ", model.score(bow_val, labels_val))

if __name__=="__main__":
    main()