from dense_sampling import *
from sklearn.metrics import classification_report
import numpy as np
import pickle
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import GridSearchCV


def load_dataset(path):
    try:
        with open('data/dataset.pkl', 'rb') as f:
            dataset, labels = pickle.load(f)  # Assegurem que es carreguen tant dataset com labels
    except:
        dataset = []
        labels = []
        for root, dirs, files in os.walk(path):
            for dir_name in dirs[:2]:  # Només agafa els primers 3 directoris
                folder_path = os.path.join(root, dir_name)
                
                for file in os.listdir(folder_path)[:200]:  # Només agafa les primeres 200 imatges
                    img_path = os.path.join(folder_path, file)
                    img = cv2.imread(img_path)

                    if img is not None:
                        img_resized = img
                        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                        dataset.append((img_path, img_gray))
                        labels.append(dir_name)
        # Guarda tant dataset com labels al pickle
        with open('data/dataset.pkl', 'wb') as f:
            pickle.dump((dataset, labels), f)
    return dataset, labels

def preprocessar_imatge(quadrant, llindar):
    return np.var(quadrant)>=llindar

# imatge = cv2.imread('data/Cervical_Cancer/cervix_dyk/cervix_dyk_0032.jpg')
# imatge = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)

# w, h = imatge.shape
# quadrants = [imatge[0:h//2, 0:w//2], imatge[0:h//2, w//2:w], imatge[h//2:h, 0:w//2], imatge[h//2:h, w//2:w]]

# for i, quadrant in enumerate(quadrants):
#     cv2.imshow(f'Quadrant {i+1}', quadrant)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for quadrant in quadrants:
#     print(preprocessar_imatge(quadrant, 90))

def identifica_quadrant(X, y, llindar=120): #Això realment pot anar a load_dataset
    quadrants = {}
    labels = {}
    y_encoded = encode_labels(y)
    # print(X.shape)
    for (nom, imatge), label in zip(X, y_encoded):
        # print(imatge.shape)
        h, w = imatge.shape
        divisio = [imatge[0:h//2, 0:w//2], imatge[0:h//2, w//2:w], imatge[h//2:h, 0:w//2], imatge[h//2:h, w//2:w]]
        quadrants[nom] = []
        for quad in divisio:
            if preprocessar_imatge(quad, llindar):
                quadrants[nom].append(quad)
                labels[nom] = label
        
    # print(quadrants)
    
    # labels_encoded = encode_labels(labels)

    return quadrants, labels


def predir_imatges(X, model):
    # X = [[bow1, bow2, bow3, bow4], [bow1, bow2, bow3, bow4], ...]
    llista_labels = []
    for bow_img in X:
        prediccio = []
        for bow_quad in bow_img:
            pred = model.predict(bow_quad.reshape(1, -1))
            prediccio.append(pred)
        # llista_labels.append(max(prediccio, key=prediccio.count))
        llista_labels.append(max(prediccio))
   
    # print(label)
    return llista_labels

def train_test(dataset, labels, test_size=0, val_size=0.2):
    imatges = []
    labels_list = []
    for key in dataset.keys():
        labels_list.append(labels[key]) #si llindar és molt alt pot ser que no hi hagi cap imatge. Falta Controlar error.
        imatges.append(dataset[key])
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(imatges, labels_list, test_size=val_size+test_size, random_state=42)

    # X_test, X_val, y_test, y_val = train_test_split(X_val_test, y_val_test, test_size=test_size/(test_size + val_size), random_state=42)
    X_val, y_val = X_val_test, y_val_test

    return X_train, y_train, X_val, y_val, 0, 0


def extract_sift_features(images, labels, n, mask=None):
    sift = cv2.SIFT_create(nfeatures=n)
    vector = []
    categories = defaultdict(list) #Diccionari de llistes a on guardem la categoria de cada feature
    labels_nou = []
    for image, label in zip(images, labels):
        llista_img = []
        for quad in image: #Pendent de decidir les variables
            _ , descriptors = sift.detectAndCompute(quad, mask=mask)
            if descriptors is not None:
                vector.extend(descriptors)
                llista_img.append(descriptors)
        if llista_img:
            categories[label].append(llista_img)
            labels_nou.append(label)

    vector = np.array(vector)
    return vector, categories, labels_nou

def train_visual_words(vector_features, n_clusters=1024):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # desc = list(features.values())
    # values_array = np.concatenate(desc, axis=0)

    kmeans.fit(np.array(vector_features))
    return kmeans

def bag_of_words_histogram(vector, features, n_clusters=1024):
    # Create the bag of words
    #dubte sobre si bow ha de ser una matriu o ha de ser una llista de matrius unidimensionals
    bow = []
    kmeans = train_visual_words(vector, n_clusters)
    # bow = np.zeros((len(features), kmeans.n_clusters))
    for label, image_feature in features.items():
        for image in image_feature:
            llista_bows = []
            for descriptor in image: #descriptors de la imatge
                hist_label = np.zeros(shape = kmeans.n_clusters) 
                pred = kmeans.predict(descriptor) #predim a quin cluster correspon cada descriptor
                for i in pred:
                    hist_label[i] += 1 #comptem quants descriptors tenim a cada cluster
                llista_bows.append(hist_label)
            bow.append(llista_bows)
    print(len(bow))   
    # bow = np.array(bow)
    return bow

def grid_search(X_train, y_train, model, parameters):
    grid_search_cv = GridSearchCV(estimator=model, param_grid=parameters, cv=10, n_jobs=8) #Buscarà els millors paràmetres
    # print(X_train.shape, y_train.shape)
    grid_search_cv.fit(X_train, y_train)
    return grid_search_cv.best_params_, grid_search_cv.best_estimator_

def train_logistic_regression(X_train, y_train, c=0.1, solver="newton-cg", max_iter=5000, penalty="l2", classificador="ovr"):
    llista_quads = []
    llista_labels = []
    for train, label in zip(X_train, y_train):
        for j in train:
            llista_quads.append(j)
            llista_labels.append(label)
    X_train = np.array(llista_quads)
    y_train = np.array(llista_labels)

    c_values = [0.01, 0.1, 0.5, 0.75]
    # c_values = [0.1]
    parameters = {'C': c_values, 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], 'max_iter': [1000, 2500, 5000], 'penalty':["l2"]}
    # parameters = {'C': c_values, 'solver': ['liblinear'], 'max_iter': [1000], 'penalty':["l2"]}
    lr = linear_model.LogisticRegression(random_state=42)
    best_params, model = grid_search(X_train, y_train, lr, parameters)
    model = OneVsRestClassifier(model) if classificador == "ovr" else OneVsOneClassifier(model)
    model.fit(X_train, y_train)

    return model, best_params