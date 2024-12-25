from load_dataset import *
from sift import *
from dense_sampling import *
from sklearn import linear_model, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import GridSearchCV


def grid_search(X_train, y_train, model, parameters):
    grid_search_cv = GridSearchCV(estimator=model, param_grid=parameters, cv=3, n_jobs=8) #Buscarà els millors paràmetres
    # print(X_train.shape, y_train.shape)
    print(X_train, y_train)
    grid_search_cv.fit(X_train, y_train)
    return grid_search_cv.best_params_, grid_search_cv.best_estimator_

def train_logistic_regression(X_train, y_train, c=0.1, solver="newton-cg", max_iter=5000, penalty="l2", classificador="ovr"):
    # c_values = [0.1, 0.5, 0.75]
    c_values = [0.1]
    # parameters = {'C': c_values, 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], 'max_iter': [1000, 2500, 5000], 'penalty':["l2"]}
    parameters = {'C': c_values, 'solver': ['liblinear'], 'max_iter': [1000], 'penalty':["l2"]}
    lr = linear_model.LogisticRegression(random_state=42)
    best_params, model = grid_search(X_train, y_train, lr, parameters)
    model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
    model.fit(X_train, y_train)

    return model, best_params

# def train_svc(X_train, y_train, c=1.0, kernel="rbf", classificador="ovr"): 
#     #kernel: "linear", "poly", "rbf", "sigmoid" 
#     # classificador: "ovr", "ovo"
#     svc = svm.SVC(C=c, kernel=kernel, decision_function_shape=classificador)
#     svc.fit(X_train, y_train)
#     return svc

def train_svc(bow, y_train, c=1.0, kernel="sigmoid", classificador="ovr"):
    parameters = {'C': [0.1, 0.5, 0.75], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    sv = svm.SVC(random_state=42, probability=True)
    best_params, model = grid_search(bow, y_train, sv, parameters)

    model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
    model.fit(bow, y_train)
    return model, best_params

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, classificador="ovr"):
    # n_estimators = [100, 200, 300]
    n_estimators = [100]
    # parameters = {'n_estimators': n_estimators, 'max_depth': [None, 10, 20, 30]}
    parameters = {'n_estimators': n_estimators, 'max_depth': [None]}
    rf = RandomForestClassifier(random_state=42)
    best_params, model = grid_search(X_train, y_train, rf, parameters)
    model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
    model.fit(X_train, y_train)
    return model, best_params


def prediccio_tests(model, X_test, y_test):
    prediccio = model.predict(X_test)
    results = {}
    results["train_accuracy"] = accuracy_score(y_test, prediccio)
    results["train_precision"] = precision_score(y_test, prediccio)
    results["train_recall"] = recall_score(y_test, prediccio)
    results["train_f1"] = f1_score(y_test, prediccio)

    return prediccio, results

def main():
    print("Carregant dataset...")
    data, labels = load_dataset('data/Cervical_Cancer')
    print(f"Dataset carregat amb {len(data)} classes.")

    print("Dividint el dataset en entrenament, validació i test...")
    X_train, y_train, X_val, y_val, X_test, y_test = train_test(data, labels)
    print(f"Dataset dividit: {len(X_train)} entrenament, {len(X_val)} validació, {len(X_test)} test.")

    print("Entrenant i validant els models...")

    models = ["train_linear_regression", "train_logistic_regression", "train_svc", "random_forest"]
    for m in models:
        print(f"\n=== Entrenant {m} ===")
        try:
            print("Entrenant el model...")
            model = eval(m)(X_train, y_train)  #Es queda encallat i no acaba
            print(f"Model entrenat: {model}")

            print("Validant el model...")
            val_score = validation(model, X_val, y_val)
            print(f"Validació {m}: {val_score:.4f}")

            print("Prediccions i càlcul de mètriques...")
            preds, results = prediccio_tests(model, X_test, y_test)
            print(f"Resultats per {m}: {results}")
        except Exception as e:
            print(f"Error durant l'entrenament/validació de {m}: {e}")

if __name__ == "__main__":
    # main()
    pass