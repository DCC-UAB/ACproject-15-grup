import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report
import pandas as pd
from load_dataset import *
from dense_sampling import *
from sift import *
from bow import *
from train_test import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc



def get_metrics(model, X_val, y_val):
    predictions = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, average='weighted')
    recall = recall_score(y_val, predictions, average='weighted')
    f1 = f1_score(y_val, predictions, average='weighted')

    return accuracy, precision, recall, f1


def show_metrics(models, y_val):
    table = []
    for tipus_model in models:
        for m in tipus_model:
            accuracy, precision, recall, f1 = get_metrics(m[0][0], m[-1], y_val)
            if str(m[0][0].estimator).startswith("LogisticRegression"):
                table.append({'model': m[0][0].estimator, 'n_features': m[1], 'n_clusters': m[2], 
                            'c': m[0][1]['C'], 'max_iter': m[0][1]["max_iter"], 'penalty': m[0][1]["penalty"], 
                            "solver": m[0][1]["solver"], 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})
            else:
                table.append({'model': m[0][0].estimator, 'n_features': m[1], 'n_clusters': m[2], 
                            'c': m[0][1]['C'], 'kernel': m[0][1]["kernel"], 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

    df = pd.DataFrame(table)
    df = round(df, 2)
    # df = df.sort_values(by=['accuracy', 'precision', 'recall', 'f1'], ascending=False)
    return df

def execute_models():
    sift = True
    print("Carregant i processant el dataset...")
    dataset_path = 'data/Cervical_Cancer'
    data, labels = load_dataset(dataset_path)
    labels_encoded = encode_labels(labels)
    X_train, y_train, X_val, y_val, X_test, y_test = train_test(data, labels_encoded)
    # classes = list(set(labels))
    features = [128]
    n_clusters = [1024, 2028]
    dict_bows_train = {}
    dict_bows_val = {}
    # llista_bows_test = []
    passa = [10, 15, 20]
    width = [2, 5, 10]
    models = [[], []] # logistic_regression, svc


    print("Extracció de característiques SIFT i creant histograma BoW...")
    if sift:
        try:
            with open("data/bow_sift_train.pkl", 'rb') as f:
                dict_bows_train = pickle.load(f)
            with open("data/bow_sift_val.pkl", 'rb') as f:
                dict_bows_val = pickle.load(f)
            # with open("data/bow_sift_test.pkl", 'rb') as f:
            #     bow_test = pickle.load(f)
        except:
            for i in features:
                print("Feature number: ", i)
                vectors_train, features_train = extract_sift_features(X_train, y_train, i, None)
                vectors_val, features_val = extract_sift_features(X_val, y_val, i, None)
                # vectors_test, features_test = extract_sift_features(X_test, y_test, i, None)
                dict_bows_train[i] = []
                dict_bows_val[i] = []
                for j in n_clusters:
                    print("Number of clusters: ", j)
                    bow_train = bag_of_words_histogram(vectors_train, features_train, n_clusters=j)
                    bow_val = bag_of_words_histogram(vectors_val, features_val, n_clusters=j)
                    # bow_test = bag_of_words_histogram(vectors_test, features_test, n_clusters=j)
                    dict_bows_train[i].append(bow_train)
                    dict_bows_val[i].append(bow_val)
                    # llista_bows_test.extend(bow_test)

            with open("data/bow_sift_train.pkl", 'wb') as f:
                pickle.dump(dict_bows_train, f)
            with open("data/bow_sift_val.pkl", 'wb') as f:
                pickle.dump(dict_bows_val, f)
            # with open("data/bow_sift_test.pkl", 'wb') as f:
            #     pickle.dump(llista_bows_test, f)

    else:
        try:
            with open("data/bow_dense_train.pkl", 'rb') as f:
                dict_bows_train = pickle.load(f)
            with open("data/bow_dense_val.pkl", 'rb') as f:
                dict_bows_val = pickle.load(f)
            # with open("data/bow_dense_test.pkl", 'rb') as f:
            #     bow_test = pickle.load(f)
        except:
            for i in features:
                print("Feature number: ", i)
                for j in passa:
                    for w in width:
                        vectors_train, features_train = dense_sampling(X_train, y_train, j, w, i)
                        vectors_val, features_val = dense_sampling(X_val, y_val, j, w, i)
                        # vectors_test, features_test = dense_sampling(X_test, y_test, j, w, i)
                        dict_bows_train[i] = []
                        dict_bows_val[i] = []
                        for k in n_clusters:
                            print("Number of clusters: ", k)
                            bow_train = bag_of_words_histogram(vectors_train, features_train, n_clusters=k)
                            bow_val = bag_of_words_histogram(vectors_val, features_val, n_clusters=k)
                            # bow_test = bag_of_words_histogram(vectors_test, features, n_clusters=k)
                            dict_bows_train[i].append(bow_train)
                            dict_bows_val[i].append(bow_val)
                            # llista_bows_test.extend(bow_test)

    for feature, ll_bow_train in dict_bows_train.items():
        for index in range(len(ll_bow_train)):
            # print(ll_bow_train[index])
            models[0].append([train_logistic_regression(ll_bow_train[index], y_train), feature, n_clusters[index], dict_bows_val[feature][index]])
            models[1].append([train_svc(ll_bow_train[index], y_train), feature, n_clusters[index], dict_bows_val[feature][index]])
    # print(models)
    return models, y_val


def show_confusion_matrix(models, X_val, y_val):
    for tipus_model in models:
        for m in tipus_model:
            y_pred = m[0].predict(X_val)
            cm = confusion_matrix(y_val, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["1","2", "3"])
            disp.plot()
            plt.title(f'Confusion Matrix for {m[4]} and C={m[1]}')
            plt.show()


def show_roc_curve(model, X_val, y_val):
    n_classes = len(set(y_val))
    y_val_bin = label_binarize(y_val, classes=range(n_classes))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    plt.figure(figsize=(10, 8))
    roc_auc = []
    y_pred = model.predict_proba(X_val)
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_val_bin[:, i], y_pred[:, i])
        auc_score = auc(fpr, tpr)
        roc_auc.append(auc_score)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Clase {i} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Aleatorio (AUC = 0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC para cada categoría')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def visualize_bow_histogram(bow, labels, cluster_names=None):
    """
    Visualitza un histograma BoW (Bag of Words).
    
    :param bow: array numpy amb els histogrames (files: imatges, columnes: clústers)
    :param labels: array numpy amb les etiquetes corresponents a cada histograma
    :param cluster_names: llista opcional amb els noms dels clústers
    """
    num_clusters = bow.shape[1]
    num_images = bow.shape[0]
    colors=None
    mean_histogram = np.mean(bow, axis=0)

    if cluster_names is None:
        cluster_names = [f"Cluster {i}" for i in range(num_clusters)]
    if colors is None:
        colors = [plt.cm.tab20(i / num_clusters) for i in range(num_clusters)]

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_clusters), mean_histogram, tick_label=cluster_names, color=colors)
    plt.xlabel("Visual Words")
    plt.ylabel("Freqüència mitjana")
    plt.title("Histograma BoW (Bag of Words)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    model = "xgboost" #"logistic", "svc", "random_forest", "xgboost"
    n_clusters = 64
    num_dades = 300
    num_directoris = 2
    test_size = 0.2
    val_size = 0
    class_multiclass = "ovr" #"ovr", "ovo"
  
    print("Carregant i processant el dataset...")
    dataset_path = 'data/Cervical_Cancer'
    data, labels = load_dataset(dataset_path, num_dades, num_directoris)
    labels_encoded = encode_labels(labels)
    X_train, y_train, X_test, y_test, _, _ = train_test(data, labels_encoded, test_size=test_size, val_size=val_size)
    print("Extracció de característiques SIFT i creant histograma BoW...")
    print("Creant els BoW...")
    sift  = cv2.xfeatures2d.SIFT_create()

    vectors_train, train_features = extract_sift_features(sift, X_train, y_train)
    _, test_features = extract_sift_features(sift, X_test, y_test)

    kmeans = train_visual_words(vectors_train, n_clusters)

    bow_train, labels_train = bag_of_words_histogram(train_features, kmeans)
    bow_test, labels_test = bag_of_words_histogram(test_features, kmeans)

    # visualize_bow_histogram(bow_train, labels_train)
    model, best_params = train_xgboost(bow_train, labels_train, class_multiclass)
    show_roc_curve(model, bow_test, labels_test)
        

if __name__ == '__main__':
    main()
    pass