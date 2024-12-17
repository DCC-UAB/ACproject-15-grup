from train_test import train_logistic_regression, train_svc
from load_dataset import load_dataset, train_test, encode_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from sift import extract_sift_features
from bow import bag_of_words_histogram
from dense_sampling import dense_sampling


def one_vs_all(X_train, y_train, X_val, y_val, X_test, y_test, classes, model_fn):
    # Convertim les imatges a vectors per no tenir problemes al executar
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    models = {}
    metrics = {}

    for i, classe in enumerate(classes):
        print(f"\nEntrenant model One-vs-All per la classe: {classe}")
        # Creem etiquetes binàries: 1 per la classe actual, 0 per totes les altres
        y_train_bin = (y_train == i).astype(int)
        y_val_bin = (y_val == i).astype(int)
        y_test_bin = (y_test == i).astype(int)

        model = model_fn(X_train, y_train_bin)
        models[classe] = model

        # Validem i calculem les mètriques
        y_pred = model.predict(X_val)
        metrics[classe] = {
            "accuracy": accuracy_score(y_val_bin, y_pred),
            "precision": precision_score(y_val_bin, y_pred, zero_division=0),
            "recall": recall_score(y_val_bin, y_pred, zero_division=0),
            "f1": f1_score(y_val_bin, y_pred, zero_division=0),
        }
        print(f"Mètriques per {classe}: {metrics[classe]}")

    test_metrics = {}
    for classe, model in models.items():
        y_test_bin = (y_test == classes.index(classe)).astype(int)
        y_pred_test = model.predict(X_test)
        test_metrics[classe] = {
            "accuracy": accuracy_score(y_test_bin, y_pred_test),
            "precision": precision_score(y_test_bin, y_pred_test, zero_division=0),
            "recall": recall_score(y_test_bin, y_pred_test, zero_division=0),
            "f1": f1_score(y_test_bin, y_pred_test, zero_division=0),
        }
        print(f"Test Mètriques per {classe}: {test_metrics[classe]}")

    return models, metrics, test_metrics

def plot_metrics(metrics, metric_name, title):
    classes = list(metrics.keys())
    values = [metrics[classe][metric_name] for classe in classes]

    plt.figure(figsize=(10, 6))
    plt.bar(classes, values, alpha=0.7, edgecolor='black')
    plt.xlabel("Classes")
    plt.ylabel(metric_name.capitalize())
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_all_metrics(metrics, title_prefix="Validation"):
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_metrics(metrics, metric, f"{title_prefix} {metric.capitalize()} per classe")


def main():
    sift = True
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
            # with open("data/bow_sift_test.pkl", 'rb') as f:
            #     bow_test = pickle.load(f)
        except:
            vectors, features = extract_sift_features(X_train, y_train, 128, None)
            print(len(vectors), len(features))
            bow_train = bag_of_words_histogram(vectors, features)
            # print(bow_train)
            vectors, features = extract_sift_features(X_val, y_val, 128, None)
            bow_val = bag_of_words_histogram(vectors, features)
            # vectors, features = extract_sift_features(X_test, y_test, 128, None)
            # bow_test = bag_of_words_histogram(vectors, features, sift=True, fase="test")
    else:
        try:
            with open("data/bow_dense_train.pkl", 'rb') as f:
                bow_train = pickle.load(f)
            with open("data/bow_dense_val.pkl", 'rb') as f:
                bow_val = pickle.load(f)
            with open("data/bow_dense_test.pkl", 'rb') as f:
                bow_test = pickle.load(f)
        except:
            vectors, features = dense_sampling(X_train, y_train, 15, 5, 128)
            bow_train = bag_of_words_histogram(vectors, features, sift=False, fase="train")
            vectors, features = dense_sampling(X_val, y_val, 15, 5, 128)
            bow_val = bag_of_words_histogram(vectors, features, sift=False, fase="val")
            # vectors, features = dense_sampling(X_test, y_test, 15, 5, 128)
            # bow_test = bag_of_words_histogram(vectors, features, sift=False, fase="test")


    # # Entrenem el model One-vs-All amb Logistic Regression
    # print("\n=== Logistic Regression ===")
    # models_lr, val_metrics_lr, test_metrics_lr = one_vs_all(
    #     bow_train, y_train, bow_val, y_val, bow_test, y_test, classes, train_logistic_regression
    # )

    # # Entrenem el model One-vs-All amb SVC
    # print("\n=== Support Vector Classifier ===")
    # models_svc, val_metrics_svc, test_metrics_svc = one_vs_all(
    #     X_train, y_train, X_val, y_val, X_test, y_test, classes, train_svc
    # )
    # print([max(x) for x in bow_train], len(y_train))
    # models = train_svc(bow_train, y_train)
    # for model in models:
    #     print(model[0].score(bow_val, y_val), model[1])
    print(len(bow_train), len(y_train))
    model = train_logistic_regression(bow_train, y_train)
    print(model[0].score(bow_train, y_train))
    print(model[0].score(bow_val, y_val), model[1])

    # Dibuixar un gràfic amb els punts i la classificacio segons SVC
    # plt.figure(figsize=(10, 6))
    # plt.scatter(bow_val[:, 0], bow_val[:, 1], c=models[0][0].predict(bow_val), cmap='viridis', alpha=0.7)
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    # plt.title("Visualització de les classes amb SVC")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()


    # print("Generant gràfics de validació per Logistic Regression...")
    # plot_all_metrics(val_metrics_lr, title_prefix="Validation (Logistic Regression)")

    # # Plota les mètriques de test:
    # print("Generant gràfics de test per Logistic Regression...")
    # plot_all_metrics(test_metrics_lr, title_prefix="Test (Logistic Regression)")

    # # Plota les mètriques de validació per SVC:
    # print("Generant gràfics de validació per SVC...")
    # plot_all_metrics(val_metrics_svc, title_prefix="Validation (SVC)")

    # # Plota les mètriques de test per SVC:
    # print("Generant gràfics de test per SVC...")
    # plot_all_metrics(test_metrics_svc, title_prefix="Test (SVC)")

    # # Guardem els models en fitxers dins de /data
    # models_dir = os.path.join("data", "models")
    # os.makedirs(models_dir, exist_ok=True)

    # for classe, model in models_lr.items():
    #     with open(os.path.join(models_dir, f"logistic_regression_{classe}.pkl"), 'wb') as f:
    #         pickle.dump(model, f)
    
    # for classe, model in models_svc.items():
    #     with open(os.path.join(models_dir, f"svc_{classe}.pkl"), 'wb') as f:
    #         pickle.dump(model, f)

    # print("\nModels guardats a la carpeta '/data/models'.")

if __name__ == "__main__":
    main()
