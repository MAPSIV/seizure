import os
import sklearn
import sklearn.model_selection
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import optuna

path_features  = "/export/home/mapiv02/seizure/features/"
path_metadatos = "/export/home/mapiv02/seizure/datos_procesados/"
save_path = "/export/home/mapiv02/seizure/"
N_datos_clase = 40000

first = True
for filename in os.listdir(path_features):
    if filename.endswith("_windows_features.npz"):
        metrics_windows = np.load(os.path.join(path_features, filename))["features"]
        metadatos = pd.read_parquet(os.path.join(path_metadatos, filename[:-20]+"metadatos.parquet"))
        X_aux = np.array(metrics_windows)
        y_aux = metadatos["tag"]
        X_aux = np.reshape(X_aux,(X_aux.shape[0],X_aux.shape[1]*X_aux.shape[2]))
        if first:
            first = False
            X = X_aux
            y = y_aux
        else:
            X = np.concatenate((X, X_aux))
            y = np.concatenate((y, y_aux))

print(X.shape)
print(y.shape)
X_class1 = X[y==1][:N_datos_clase]
X_class0 = X[y==0][:N_datos_clase]
y_class1 = y[y==1][:N_datos_clase]
y_class0 = y[y==0][:N_datos_clase]
X = np.concatenate((X_class0, X_class1))
y = np.concatenate((y_class0, y_class1))
print(X.shape)
print(y.shape)

def objective(trial):
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        X, y, test_size=0.25, random_state=0
    )    
    Cs = trial.suggest_float('C', 0.1, 100)
    kernels = trial.suggest_categorical("kernel", ["poly", "rbf", "sigmoid"])
    degrees = trial.suggest_int('degree', 2, 10)
    gammas = trial.suggest_categorical("gamma", ["scale", "auto"])
    shrinkings = trial.suggest_categorical("shrinking", [True, False])
    probabilities = trial.suggest_categorical("probability", [True, False])

    svm=SVC(C=Cs, kernel=kernels, degree=degrees, gamma=gammas, shrinking=shrinkings, 
            probability=probabilities, random_state=0)

    svm.fit(train_x, train_y)

    preds = svm.predict(valid_x)

    accuracy = sklearn.metrics.accuracy_score(valid_y, preds)
    f1 = sklearn.metrics.f1_score(valid_y, preds)
    precision = sklearn.metrics.precision_score(valid_y, preds)
    recall = sklearn.metrics.recall_score(valid_y, preds)
    roc = sklearn.metrics.roc_auc_score(valid_y, preds)
    
    return accuracy, f1, precision, recall, roc

sampler = optuna.samplers.NSGAIISampler()
study = optuna.create_study(sampler=sampler, directions=['maximize', 'maximize', 'maximize', 'maximize', 'maximize'])

study.optimize(objective, n_trials=1)
trials = study.best_trials

dic_trials = {"accuracy":[], "f1":[], "precision":[], "recall":[], "roc":[], "C":[], "kernel":[], "degree":[], "gamma":[], "shrinking":[], "probability":[]}
for trial in trials:
    dic_trials["accuracy"].append(trial.values[0])
    dic_trials["f1"].append(trial.values[1])
    dic_trials["precision"].append(trial.values[2])
    dic_trials["recall"].append(trial.values[3])
    dic_trials["roc"].append(trial.values[4])

    dic_trials["C"].append(trial.params["C"])
    dic_trials["kernel"].append(trial.params["kernel"])
    dic_trials["degree"].append(trial.params["degree"])
    dic_trials["gamma"].append(trial.params["gamma"])
    dic_trials["shrinking"].append(trial.params["shrinking"])
    dic_trials["probability"].append(trial.params["probability"])

df_trials = pd.DataFrame(dic_trials)
df_trials.to_parquet(os.path.join(save_path, "best_parameters_small.parquet"))