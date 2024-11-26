from load_dataset import *
from sklearn import linear_model, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecissionTreeClassifier
from sklearn.model_selection import GridSearchCV



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

def random_forest(X_train, y_train):
    rf = DecissionTreeClassifier(random_state=42)
    
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
        clf = DecissionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
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
    train_set, val_set, test_set = train_test(data)
    
    #Falta funció per separar X, y

    # X_train, y_train = load_images(train_set)
    # X_val, y_val = load_images(val_set)
    # X_test, y_test = load_images(test_set)
    
    # model = train_linear_regression(X_train, y_train)
    # model = train_logistic_regression(X_train, y_train)
    # model = train_svc(X_train, y_train)
    # model = random_forest(X_train, y_train)
    # model = tree_pruning(model, X_train, y_train)
    
    # validation(model, X_val, y_val)
    # prediccio_tests(model, X_test, y_test)