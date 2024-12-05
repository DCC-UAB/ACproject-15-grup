from load_dataset import *
from sift import *
from dense_sampling import *
from sklearn import linear_model, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from sklearn.multiclass import OneVsRestClassifier


def train_linear_regression(X_train, y_train):
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)

    return lr

def train_logistic_regression(X_train, y_train, c=10):
    lr = linear_model.LogisticRegression(C=c, solver="liblinear", max_iter=5000, penalty="l2")
    lr.fit(X_train, y_train)
    return lr

# def train_svc(X_train, y_train, c=1.0, kernel="rbf", classificador="ovr"): 
#     #kernel: "linear", "poly", "rbf", "sigmoid" 
#     # classificador: "ovr", "ovo"
#     svc = svm.SVC(C=c, kernel=kernel, decision_function_shape=classificador)
#     svc.fit(X_train, y_train)
#     return svc

def train_svc(bow, y_train):
    models = []
    for i in [0.1, 0.5, 1, 5, 10, 20, 40, 100, 1000]:
        clf = OneVsRestClassifier(svm.SVC(C=i, kernel="sigmoid", random_state=42)).fit(bow, y_train)
        models.append((clf, i))
    return models

def random_forest(X_train, y_train):
    rf = DecisionTreeClassifier(random_state=42)
    
    #Això no sé segur
    # parameters= {'criterion':['entropy', 'gini'],'max_depth' : [2,4,6,8,10,12]  ,'splitter':["best","random"],'min_samples_split':[2,3,4,6,7]}

    # grid_search_cv = GridSearchCV(estimator=rf, param_grid=parameters, cv=3, n_jobs=14) #Buscarà els millors paràmetres
    # grid_search_cv.fit(X_train, y_train)
    
    rf.fit(X_train, y_train)
    return rf

def tree_pruning(rf, X_train, y_train):
    path = rf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    return clfs #Faltaria analitzar i agafar el millor arbre

def validation(model, X_val, y_val):
    return model.score(X_val, y_val)


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
    main()