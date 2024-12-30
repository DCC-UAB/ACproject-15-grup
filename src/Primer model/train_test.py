from load_dataset import *
from sift import *
from dense_sampling import *
from sklearn import linear_model, svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


def grid_search(X_train, y_train, model, parameters):
    """
    Mètode que busca els millors paràmetres per un model donat.

    :param X_train: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param model: model a entrenar
    :param parameters: diccionari amb els paràmetres a buscar
    :return: millors paràmetres i millor model
    """
    grid_search_cv = GridSearchCV(estimator=model, param_grid=parameters, cv=3, n_jobs=8) #Buscarà els millors paràmetres
    grid_search_cv.fit(X_train, y_train)
    return grid_search_cv.best_params_, grid_search_cv.best_estimator_

def train_logistic_regression(X_train, y_train, c=0.1, solver="newton-cg", max_iter=5000, penalty="l2", classificador="ovr"):
    """
    Mètode que entrena un model de regressió logística amb els paràmetres donats.

    :param X_train: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param c: float amb el valor de C
    :param solver: string amb el tipus de solver
    :param max_iter: int amb el nombre màxim d'iteracions
    :param penalty: string amb el tipus de penalització
    :param classificador: string amb el tipus de classificador
    :return: model de regressió logística entrenat
    
    """
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    # c_values = [0.1]
    parameters = {'C': c_values, 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'saga'], 'penalty':["l2"]}
    # parameters = {'C': c_values, 'solver': ['liblinear'], 'max_iter': [1000], 'penalty':["l2"]}
    lr = linear_model.LogisticRegression(random_state=42)
    best_params, model = grid_search(X_train, y_train, lr, parameters)
    model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
    model.fit(X_train, y_train)

    return model, best_params

def train_svc(bow, y_train, c=1.0, kernel="sigmoid", classificador="ovr"):
    """
    Mètode que entrena un model SVC amb els paràmetres donats.

    :param bow: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param c: float amb el valor de C
    :param kernel: string amb el tipus de kernel
    :param classificador: string amb el tipus de classificador
    :return: model SVC entrenat
    """
    # parameters = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], "decision_function_shape": ['ovo', 'ovr']}
    parameters = {'C': [1], 'kernel': ['rbf'], "decision_function_shape": ['ovr']}

    sv = svm.SVC(random_state=42, probability=True)
    best_params, model = grid_search(bow, y_train, sv, parameters)

    model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
    model.fit(bow, y_train)
    return model, best_params

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, classificador="ovr"):
    """
    Mètode que entrena un model Random Forest amb els paràmetres donats.

    :param X_train: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param n_estimators: int amb el nombre d'estimadors
    :param max_depth: int amb la profunditat màxima
    :param classificador: string amb el tipus de classificador
    :return: model Random Forest entrenat
    """
    parameters = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30], "criterion": ['gini', 'entropy'], "bootstrap": [True, False]}
    # parameters = {'n_estimators': [100], 'max_depth': [None]}
    rf = RandomForestClassifier(random_state=42)
    best_params, model = grid_search(X_train, y_train, rf, parameters)
    model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
    model.fit(X_train, y_train)
    return model, best_params

def train_xgboost(X_train, y_train, classificador="ovr"):
    """
    Mètode que entrena un model XGBoost amb els paràmetres donats.

    :param X_train: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param classificador: string amb el tipus de classificador
    :return: model XGBoost entrenat
    """
    num_class = len(np.unique(y_train))
    # parameters = {'max_depth': [3, 6, 10], 'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'eval_metric': ['mlogloss'], "lambda": [0, 0.1, 1], "alpha": [0, 0.1, 1], "objective":["multi:softmax"]}
    parameters = {'eval_metric': ['logloss'], 'learning_rate': [0.1], 'max_depth': [3], 'n_estimators': [100]}
    xgb_model = xgb.XGBClassifier(random_state=42, num_class=num_class)
    best_params, model = grid_search(X_train, y_train, xgb_model, parameters)
    model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
    model.fit(X_train, y_train)
    return model, best_params

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