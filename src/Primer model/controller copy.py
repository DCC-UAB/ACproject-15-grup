from load_dataset import load_dataset
from load_dataset import encode_labels
from load_dataset import train_test
from sift import extract_sift_features
from bow import train_visual_words
from bow import bag_of_words_histogram
from train_test import train_logistic_regression
from train_test import train_svc
from train_test import train_random_forest
from train_test import train_xgboost
from sklearn import metrics
import cv2
from visualitzacions import get_metrics
import wandb
from dense_sampling import dense_sampling
import pickle
from visualitzacions import show_roc_curve


def main():
    wandb.login(key="e04f0be8ca1d2a05e435577f94aad6b2d9ad31bb")
    # models = ["logistic", "svc", "random_forest", "xgboost"]
    models = ["xgboost"]
    detector = "sift" #"dense"
    n_clusters = [32, 64, 128, 256, 512, 1024]
    num_dades = 5000
    num_directoris = 5
    # pases = 15
    # amplada_punt = 5
    test_size = 0.2
    val_size = 0

  
    print("Carregant i processant el dataset...")
    dataset_path = 'data/Cervical_Cancer'
    data, labels = load_dataset(dataset_path, num_dades, num_directoris)
    labels_encoded = encode_labels(labels)
    X_train, y_train, X_test, y_test, _, _ = train_test(data, labels_encoded, test_size=test_size, val_size=val_size)
    
    print("Extreient característiques SIFT")
    if detector == "sift":
        sift  = cv2.xfeatures2d.SIFT_create()
        for cluster in n_clusters:
            for model_ in models:
                
                wandb.init(project="Practica2_SIFT",
                    config={
                            "model": model_,
                            "features": "SIFT",
                            "n_clusters": cluster,
                            "num_dades": num_dades,
                            "num_directoris": num_directoris,
                            "test_size": test_size,
                            "val_size": val_size
                    })

                try:
                    with open("data/bow_sift_train.pkl", 'rb') as f:
                        bow_train, labels_train = pickle.load(f)
                    with open("data/bow_sift_test.pkl", 'rb') as f:
                        bow_test, labels_test = pickle.load(f) 
                except:
                    vectors_train, train_features = extract_sift_features(sift, X_train, y_train)
                    _, val_features = extract_sift_features(sift, X_test, y_test)
                    print("Característiques SIFT extretes\n")
                    print("Creant els BoW's...")
                    kmeans = train_visual_words(vectors_train, cluster)

                    bow_train, labels_train = bag_of_words_histogram(train_features, kmeans)
                    bow_test, labels_test = bag_of_words_histogram(val_features, kmeans)
                    with open("data/bow_sift_train.pkl", 'wb') as f:
                        pickle.dump((bow_train, labels_train), f)
                    with open("data/bow_sift_test.pkl", 'wb') as f:
                        pickle.dump((bow_test, labels_test), f)
                print("BoW's creats\n")
                print("\nEntrenant el model...")

                if model_ == "svc":
                    model, best_params = train_svc(bow_train, labels_train)
                elif model_ == "logistic":
                    model, best_params = train_logistic_regression(bow_train, labels_train)
                elif model_ == "random_forest":
                    model, best_params = train_random_forest(bow_train, labels_train)
                elif model_ == "xgboost":
                    model, best_params = train_xgboost(bow_train, labels_train)
                print("Model entrenat\n")
                show_roc_curve(model, bow_test, labels_test)
                prediccio = model.predict(bow_test)
                print("Confusion Matrix: ",metrics.confusion_matrix(labels_test, prediccio))
                print("Accuracy: ",metrics.accuracy_score(labels_test, prediccio))

                print("Params: ", best_params)
                overfitting = model.score(bow_train, labels_train)
                # print("Resultats test = ", model.score(bow_test, labels_test))
                print("\nVisualitzant les mètriques...\n")
                accuracy, precision, recall, f1 = get_metrics(model, bow_test, labels_test)
                wandb.log({"overfitting":overfitting, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
                wandb.finish()
                print("Overfitting: ", overfitting)
                print("Accuracy: ", accuracy)
                print("Precision: ", precision)
                print("Recall: ", recall)
                print("F1: ", f1)

    # elif detector == "dense":
    #     sift  = cv2.xfeatures2d.SIFT_create()

    #     wandb.init(project="Practica2_DenseSampling",
    #            config={
    #                 "model": model,
    #                 "features": "Dense Sampling",
    #                 "pases": pases,
    #                 "amplada_punt": amplada_punt,
    #                 "n_clusters": n_clusters,
    #                 "num_dades": num_dades,
    #                 "num_directoris": num_directoris,
    #                 "test_size": test_size,
    #                 "val_size": val_size
    #            })
        
    #     vectors_train, train_features = dense_sampling(sift, X_train, y_train, pases, amplada_punt)
    #     _, val_features = dense_sampling(sift, X_test, y_test, pases, amplada_punt)

    #     kmeans = train_visual_words(vectors_train, n_clusters)
        
    #     bow_train, labels_train = bag_of_words_histogram(train_features, kmeans)
    #     bow_test, labels_test = bag_of_words_histogram(val_features, kmeans)

        # with open("data/bow_sift_train.pkl", 'wb') as f:
        #     pickle.dump(bow_train, f)
        # with open("data/bow_sift_test.pkl", 'wb') as f:
        #     pickle.dump(bow_test, f)

    # print("Entrenant el model...")
    # # print(len(bow_train), len(y_train))
    # # model, best_params = train_logistic_regression(bow_train, labels_train)
    # if model == "svc":
    #     model, best_params = train_svc(bow_train, labels_train)
    # elif model == "logistic":
    #     model, best_params = train_logistic_regression(bow_train, labels_train)
    # elif model == "random_forest":
    #     model, best_params = train_random_forest(bow_train, labels_train)
    # elif model == "xgboost":
    #     model, best_params = train_xgboost(bow_train, labels_train)
    # prediccio = model.predict(bow_test)
    # print("confusion matrix: ",metrics.confusion_matrix(labels_test, prediccio))
    # print("accuracy: ",metrics.accuracy_score(labels_test, prediccio))

    # print("Params: ", best_params)
    # overfitting = model.score(bow_train, labels_train)
    # print("Overfitting = ", overfitting)
    # print("Resultats test = ", model.score(bow_test, labels_test))
    # print("Visualitzant les mètriques...")
    # accuracy, precision, recall, f1 = get_metrics(model, bow_test, labels_test)
    # wandb.log({"overfitting":overfitting, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
    # wandb.finish()
    # print("Accuracy: ", accuracy)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1: ", f1)

if __name__=="__main__":
    main()