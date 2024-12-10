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



def get_metrics(model, X_val, y_val):
    predictions = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, average='weighted')
    recall = recall_score(y_val, predictions, average='weighted')
    f1 = f1_score(y_val, predictions, average='weighted')

    return accuracy, precision, recall, f1


def show_metrics(models, X_val, y_val):
    table = []
    for tipus_model in models:
        for m, bow in zip(tipus_model, X_val):
            accuracy, precision, recall, f1 = get_metrics(m[0], bow, y_val)
            table.append({'model': m[0].estimator, 'c': m[1]['C'], 'max_iter': m[1]["max_iter"], 'penalty': m[1]["penalty"], "solver": m[1]["solver"], 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

# Convert the list of dictionaries to a DataFrame
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
    n_clusters = [1024, 2048]
    llista_bows_train = []
    llista_bows_val = []
    # llista_bows_test = []
    passa = [10, 15, 20]
    width = [2, 5, 10]
    models = []

    print("Extracció de característiques SIFT i creant histograma BoW...")
    if sift:
        try:
            with open("data/bow_sift_train.pkl", 'rb') as f:
                bow_train = pickle.load(f)
            with open("data/bow_sift_val.pkl", 'rb') as f:
                bow_val = pickle.load(f)
            with open("data/bow_sift_test.pkl", 'rb') as f:
                bow_test = pickle.load(f)
        except:
            for i in features:
                print("Feature number: ", i)
                vectors_train, features_train = extract_sift_features(X_train, y_train, i, None)
                vectors_val, features_val = extract_sift_features(X_val, y_val, i, None)
                # vectors_test, features_test = extract_sift_features(X_test, y_test, i, None)
                for j in n_clusters:
                    print("Number of clusters: ", j)
                    bow_train = bag_of_words_histogram(vectors_train, features_train, n_clusters=j, sift=True, fase="train")
                    bow_val = bag_of_words_histogram(vectors_val, features_val, n_clusters=j, sift=True, fase="val")
                    # bow_test = bag_of_words_histogram(vectors_test, features_test, n_clusters=j, sift=True, fase="test")
                    llista_bows_train.append(bow_train)
                    llista_bows_val.append(bow_val)
                    # llista_bows_test.extend(bow_test)
                   
    else:
        try:
            with open("data/bow_dense_train.pkl", 'rb') as f:
                bow_train = pickle.load(f)
            with open("data/bow_dense_val.pkl", 'rb') as f:
                bow_val = pickle.load(f)
            with open("data/bow_dense_test.pkl", 'rb') as f:
                bow_test = pickle.load(f)
        except:
            for i in features:
                for j in passa:
                    for w in width:
                        vectors_train, features_train = dense_sampling(X_train, y_train, j, w, i)
                        vectors_val, features_val = dense_sampling(X_val, y_val, j, w, i)
                        # vectors_test, features_test = dense_sampling(X_test, y_test, j, w, i)
                        for k in n_clusters:
                            bow_train = bag_of_words_histogram(vectors_train, features_train, n_clusters=k, sift=False, fase="train")
                            bow_val = bag_of_words_histogram(vectors_val, features_val, n_clusters=k, sift=False, fase="val")
                            # bow_test = bag_of_words_histogram(vectors_test, features, n_clusters=k, sift=False, fase="test")
                            llista_bows_train.append(bow_train)
                            llista_bows_val.append(bow_val)
                            # llista_bows_test.extend(bow_test)

            # vectors, features = dense_sampling(X_test, y_test, 15, 5, 128)
            # bow_test = bag_of_words_histogram(vectors, features, sift=False, fase="test")
    for bow_train in llista_bows_train:
        print(len(bow_train), len(y_train))
        models.append([train_logistic_regression(bow_train, y_train)])

        # model2 = train_svc(bow_train, y_train) #Model es una llista amb el model i els paràmetres utilitzats
    # print(models)
    return models, llista_bows_val, y_val


def show_confusion_matrix(models, X_val, y_val):
    for tipus_model in models:
        for m in tipus_model:
            y_pred = m[0].predict(X_val)
            cm = confusion_matrix(y_val, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["1","2", "3"])
            disp.plot()
            plt.title(f'Confusion Matrix for {m[4]} and C={m[1]}')
            plt.show()



def show_roc_curve(models, X_val, y_val):
    roc_auc = []
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink', 'brown', 'black', 'yellow', 'gray']
    y_val_bin = label_binarize(y_val, classes=[0, 1, 2])  # Adjust classes as per your dataset
    for tipus_model in models:
        plt.figure()

        for index, m in enumerate(tipus_model):
            y_pred = m[0].predict_proba(X_val)
            for i in range(y_val_bin.shape[1]):
                fpr, tpr, thresholds = roc_curve(y_val_bin[:, i], y_pred[:, i])
                roc_auc.append(auc(fpr, tpr))
                plt.plot(fpr, tpr, color=colors[index], lw=2, label=f'ROC curve (AUC = {roc_auc[-1]:.2f})')
                plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

                for j, threshold in enumerate(thresholds):
                    plt.annotate(f'{threshold:.2f}', (fpr[j], tpr[j]), textcoords="offset points", xytext=(10,-10), ha='center')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {m[4]}')
        plt.legend(loc="lower right")
        plt.show()

def main():
    models, llista_bows_val, y_val = execute_models()
    print(models)
    # nom_models = ["logistic_regression", "svc"] 
    df = show_metrics(models, llista_bows_val, y_val)
    print(df)
    # show_confusion_matrix(models, bow_val, y_val)

    # show_roc_curve(models, bow_val, y_val)

if __name__ == '__main__':
    main()
    pass