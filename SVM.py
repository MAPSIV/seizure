import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import joblib

#feat_path  = "/export/home/mapiv02/seizure/features/"
#gt_path = "/export/home/mapiv02/seizure/datos_procesados/"
#save_path = "/export/home/mapiv02/seizure/"
feat_path  = "D:\\Clase\\4r\\Primer Cuatrimestre\\MAPSIV\\reto 3\\features"
gt_path = "D:\\Clase\\4r\\Primer Cuatrimestre\\MAPSIV\\reto 3\\Datos_procesados"
save_path = ""

# Load data in a list of patients
gt_list = os.listdir(gt_path)
feat_list = os.listdir(feat_path)
EEG_Feat = []
EEG_GT = []
for filename in feat_list:
    if filename.endswith("_windows_features.npz"):
        metrics_windows = np.load(os.path.join(feat_path, filename))["features"]
        metadatos = pd.read_parquet(os.path.join(gt_path, filename[:-20]+"metadatos.parquet"))['tag']
        EEG_Feat.append(np.array(metrics_windows))
        EEG_GT.append(metadatos)
        print(filename)

# Save number of patients
N_pacientes = len(EEG_GT)

# Balance data and get 1/4 of the data
for i in range(N_pacientes):
    X = EEG_Feat[i]
    y = EEG_GT[i]

    print(X.shape)
    print(y.shape)

    # Balance data
    X_class1 = X[y==1]
    X_class0 = X[y==0][:X_class1.shape[0]]
    y_class1 = y[y==1]
    y_class0 = y[y==0][:X_class1.shape[0]]

    # Get 1/4 of the data
    X_class1 = X_class1[::4]
    X_class0 = X_class0[::4]
    y_class1 = y_class1[::4]
    y_class0 = y_class0[::4]

    X = np.concatenate((X_class0, X_class1))
    y = np.concatenate((y_class0, y_class1))

    EEG_Feat[i] = X
    EEG_GT[i] = y

    print(X.shape)
    print(y.shape)



# Iteration over pacients
for i in range(N_pacientes):
    # Create model
    svm=SVC(class_weight='balanced', C=82.71567020582488, kernel='rbf', degree = 9, gamma = 'scale', shrinking = False, probability = False)
    
    # Leave one out
    X_train_pacient = EEG_Feat[:i] + EEG_Feat[i+1:]
    X_test = EEG_Feat[i]
    y_train_pacient = EEG_GT[:i] + EEG_GT[i+1:]
    y_test = EEG_GT[i]

    # Put all the training data in two arrays, one for the features and one for the labels
    X_train = X_train_pacient[0]
    y_train = y_train_pacient[0]
    for pacient in range(N_pacientes-2):
        X_train = np.concatenate((X_train, X_train_pacient[pacient]))
        y_train = np.concatenate((y_train, y_train_pacient[pacient]))
    X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
    print(i, X_train.shape)
    print(i, X_test.shape)

    # Train the model
    svm.fit(X_train, y_train)

    # Test the model
    preds = svm.predict(X_test)

    # Calculate metrics
    accuracy = sklearn.metrics.accuracy_score(y_test, preds)
    f1 = sklearn.metrics.f1_score(y_test, preds)
    precision = sklearn.metrics.precision_score(y_test, preds)
    recall = sklearn.metrics.recall_score(y_test, preds)
    roc = sklearn.metrics.roc_auc_score(y_test, preds)

    # Save metrics
    file = open(os.path.join(save_path, "statics.txt"), "a")
    file.write(str(i)+","+str(accuracy)+","+str(f1)+","+str(precision)+","+str(recall)+","+str(roc)+"\n")
    file.close()

    # Save model
    joblib.dump(svm, os.path.join(save_path, "Model_svm_"+str(i)+".joblib"))
