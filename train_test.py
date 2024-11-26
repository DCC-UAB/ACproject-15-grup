from load_dataset import *
from sklearn import linear_model, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_linear_regression(X_train, y_train):
    lr = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
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

def random_forest():
    pass

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