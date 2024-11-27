from load_dataset_copy import *
from sklearn import linear_model, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


def train_linear_regression(X_train, y_train):
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)

    return lr

def train_logistic_regression(X_train, y_train, c=10):
    lr = linear_model.LogisticRegression(C=c, solver="liblinear", max_iter=5000, penalty="l2")
    lr.fit(X_train, y_train)
    return lr

def train_svc(X_train, y_train, c=1.0, kernel="rbf", classificador="ovr"): 
    #kernel: "linear", "poly", "rbf", "sigmoid" 
    # classificador: "ovr", "ovo"
    svc = svm.SVC(C=c, kernel=kernel, decision_function_shape=classificador)
    svc.fit(X_train, y_train)
    return svc

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
    data = load_dataset('data/Cervical_Cancer')
    X_train, y_train, X_val, y_val, X_test, y_test = train_test(data)
    print(y_train)
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    models = ["train_linear_regression", "train_logistic_regression", "train_svc", "random_forest"]
    # model = train_linear_regression(X_train, y_train)
    # model = train_logistic_regression(X_train, y_train)
    # model = train_svc(X_train, y_train)
    # model = random_forest(X_train, y_train)
    # model = tree_pruning(model, X_train, y_train)
    for m in models:
        print(m)
        model = eval(m)(X_train, y_train)
        print(m)
        print(validation(model, X_val, y_val))
        print(prediccio_tests(model, X_test, y_test))
    
    # validation(model, X_val, y_val)
    # prediccio_tests(model, X_test, y_test)

if __name__ == "__main__":
    main()