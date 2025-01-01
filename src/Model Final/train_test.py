from load_dataset import *
from sift import *
from dense_sampling import *
from sklearn import linear_model, svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

def randomized_search(X_train, y_train, model, parameters, n_iter=50):
    """
    Mètode que busca els millors paràmetres per un model donat. Aquesta cerca és més ràpida que la cerca exhaustiva.

    :param X_train: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param model: model a entrenar
    :param parameters: diccionari amb els paràmetres a buscar
    :param n_iter: nombre d'iteracions
    :return: millors paràmetres i millor model
    """
    random_search_cv = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42, verbose=2)
    random_search_cv.fit(X_train, y_train)
    return random_search_cv.best_params_, random_search_cv.best_estimator_

def train_logistic_regression(X_train, y_train, c=1, solver="newton-cg", max_iter=None, penalty="l2", classificador="ovo", grid_search=False, randomized_search=False):
    """
    Mètode que entrena un model de regressió logística amb els paràmetres donats.
    Si grid_search és True o randomized_search és True, es buscaran els millors paràmetres per aquest model i els paràmetres s'han de passar com a llistes.

    :param X_train: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param c: float amb el valor de C
    :param solver: string amb el tipus de solver
    :param max_iter: int amb el nombre màxim d'iteracions
    :param penalty: string amb el tipus de penalització
    :param classificador: string amb el tipus de classificador
    :return: model de regressió logística entrenat
    
    """
    # c_values = [0.01, 0.1, 1, 10]
    # parameters = {'C': c_values, 'solver': ['liblinear'], 'max_iter': [1000], 'penalty':["l2"]}
    best_params = None
    if grid_search:
        parameters = {'C': c, 'solver': solver, 'penalty':penalty, 'max_iter':max_iter}
        lr = linear_model.LogisticRegression(random_state=42)
        best_params, model = grid_search(X_train, y_train, lr, parameters)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(X_train, y_train)
        
    if randomized_search:
        parameters = {'C': c, 'solver': solver, 'penalty':penalty, 'max_iter':max_iter}
        lr = linear_model.LogisticRegression(random_state=42)
        best_params, model = randomized_search(X_train, y_train, lr, parameters)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(X_train, y_train)
    
    if not(randomized_search and grid_search):
        model = linear_model.LogisticRegression(C=c, solver=solver, max_iter=max_iter, penalty=penalty, random_state=42)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(X_train, y_train)
    
    return model, best_params

def train_svc(bow, y_train, c=1.0, kernel="rbf", classificador="ovr", grid_search=False, randomized_search=False):
    """
    Mètode que entrena un model SVC amb els paràmetres donats.
    Si grid_search és True o randomized_search és True, es buscaran els millors paràmetres per aquest model i els paràmetres s'han de passar com a llistes.

    :param bow: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param c: float amb el valor de C
    :param kernel: string amb el tipus de kernel
    :param classificador: string amb el tipus de classificador
    :return: model SVC entrenat
    """
    # parameters = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], "decision_function_shape": ['ovo', 'ovr']}
    # parameters = {'C': [1.0], 'kernel': ["rbf"], "decision_function_shape": ["ovr"]}
    best_params = None
    if grid_search:
        parameters = {'C': c, 'kernel': kernel, "decision_function_shape": classificador}
        sv = svm.SVC(random_state=42, probability=True)
        best_params, model = grid_search(bow, y_train, sv, parameters)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(bow, y_train)
    if randomized_search:
        parameters = {'C': c, 'kernel': kernel, "decision_function_shape": classificador}
        sv = svm.SVC(random_state=42, probability=True)
        best_params, model = randomized_search(bow, y_train, sv, parameters)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(bow, y_train)

    if not(grid_search and randomized_search):
        model = svm.SVC(random_state=42, probability=True, kernel=kernel, C=c)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(bow, y_train)

    return model, best_params

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, classificador="ovr", grid_search=False, randomized_search=False):
    """
    Mètode que entrena un model Random Forest amb els paràmetres donats.
    Si grid_search és True o randomized_search és True, es buscaran els millors paràmetres per aquest model i els paràmetres s'han de passar com a llistes.

    :param X_train: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param n_estimators: int amb el nombre d'estimadors
    :param max_depth: int amb la profunditat màxima
    :param classificador: string amb el tipus de classificador
    :return: model Random Forest entrenat
    """
    # parameters = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30], "criterion": ['gini', 'entropy'], "bootstrap": [True, False]}
    # parameters = {'n_estimators': [100], 'max_depth': [None]}
    best_params = None
    if grid_search:
        parameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
        rf = RandomForestClassifier(random_state=42)
        best_params, model = grid_search(X_train, y_train, rf, parameters)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(X_train, y_train)
    if randomized_search:
        parameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
        rf = RandomForestClassifier(random_state=42)
        best_params, model = randomized_search(X_train, y_train, rf, parameters)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(X_train, y_train)

    if not(grid_search and randomized_search):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(X_train, y_train)

    return model, best_params

def train_xgboost(X_train, y_train, eval_metric="logloss", learning_rate=0.1, max_depth=3, n_estimators=100, classificador="ovr", grid_search=False, randomized_search=False):
    """
    Mètode que entrena un model XGBoost amb els paràmetres donats.
    Si grid_search és True o randomized_search és True, es buscaran els millors paràmetres per aquest model i els paràmetres s'han de passar com a llistes.

    :param X_train: np.array amb les característiques de les imatges
    :param y_train: np.array amb les etiquetes de les imatges
    :param classificador: string amb el tipus de classificador
    :return: model XGBoost entrenat
    """
    num_class = len(np.unique(y_train))
    best_params = None
    if grid_search:
        parameters = {'eval_metric': eval_metric, 'learning_rate': learning_rate, 'max_depth': max_depth, 'n_estimators': n_estimators}
        xgb_model = xgb.XGBClassifier(random_state=42, num_class=num_class)
        best_params, model = grid_search(X_train, y_train, xgb_model, parameters)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(X_train, y_train)
    if randomized_search:
        parameters = {'eval_metric': eval_metric, 'learning_rate': learning_rate, 'max_depth': max_depth, 'n_estimators': n_estimators}
        xgb_model = xgb.XGBClassifier(random_state=42, num_class=num_class)
        best_params, model = randomized_search(X_train, y_train, xgb_model, parameters)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(X_train, y_train)
    if not(grid_search and randomized_search):
        model = xgb.XGBClassifier(eval_metric=eval_metric, learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, random_state=42, num_class=num_class)
        model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
        model.fit(X_train, y_train)
    return model, best_params



if __name__ == "__main__":
    pass