import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
import pandas as pd
from load_dataset import *
from dense_sampling import *
from sift import *
from bow import *
from train_test import *


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
        for m in tipus_model:
            accuracy, precision, recall, f1 = get_metrics(m[0], X_val, y_val)
            table.append({'model': m[4], 'c': m[1], 'kernel/solver': m[2], 'classif/max_iter': m[3], 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

# Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(table)
    df = round(df, 2)
    df = df.sort_values(by=['accuracy', 'precision', 'recall', 'f1'], ascending=False)
    return df
def execute_models():
    sift = False
    print("Carregant i processant el dataset...")
    dataset_path = 'data/Cervical_Cancer'
    data, labels = load_dataset(dataset_path)
    labels_encoded = encode_labels(labels)
    X_train, y_train, X_val, y_val, X_test, y_test = train_test(data, labels_encoded)
    classes = list(set(labels))
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
            vectors, features = extract_sift_features(X_train, y_train, 128, None)
            bow_train = bag_of_words_histogram(vectors, features, sift=True, fase="train")
            vectors, features = extract_sift_features(X_val, y_val, 128, None)
            bow_val = bag_of_words_histogram(vectors, features, sift=True, fase="val")
            # vectors, features = extract_sift_features(X_test, y_test, 128, None)
            # bow_test = bag_of_words_histogram(vectors, features, sift=True, fase="test")
    else:
        try:
            with open("data/bow_dense_train.pkl", 'rb') as f:
                bow_train = pickle.load(f)
            with open("data/bow_dense_val.pkl", 'rb') as f:
                bow_val = pickle.load(f)
            # with open("data/bow_dense_test.pkl", 'rb') as f:
            #     bow_test = pickle.load(f)
        except:
            vectors, features = dense_sampling(X_train, y_train, 15, 5, 128)
            bow_train = bag_of_words_histogram(vectors, features, sift=False, fase="train")
            vectors, features = dense_sampling(X_val, y_val, 15, 5, 128)
            bow_val = bag_of_words_histogram(vectors, features, sift=False, fase="val")
            # vectors, features = dense_sampling(X_test, y_test, 15, 5, 128)
            # bow_test = bag_of_words_histogram(vectors, features, sift=False, fase="test")

    model1 = train_logistic_regression(bow_train, y_train)
    model2 = train_svc(bow_train, y_train) #Model es una llista amb el model i els paràmetres utilitzats

    return [model1, model2], bow_val, y_val

models, bow_val, y_val = execute_models()
# nom_models = ["logistic_regression", "svc"] 
df = show_metrics(models, bow_val, y_val)
print(df)

def show_confusion_matrix(models, X_val, y_val):
    for tipus_model in models:
        for m in tipus_model:
            y_pred = m[0].predict(X_val)
            cm = confusion_matrix(y_val, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["1","2", "3"])
            disp.plot()
            plt.title(f'Confusion Matrix for {m[4]} and C={m[1]}')
            plt.show()

show_confusion_matrix(models, bow_val, y_val)

from sklearn.preprocessing import label_binarize

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

# show_roc_curve(models, bow_val, y_val)
