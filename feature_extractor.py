import pandas as pd
import numpy as np
import sys
import os

path  = "/export/home/mapiv02/seizure/datos_procesados/"
save_path = "/export/home/mapiv02/seizure/features/"

for filename in os.listdir(path):
    if filename.endswith("_windows.npz"):
        metrics_windows = []
        windows = np.load(os.path.join(path, filename))
        for array_window in windows["windows"]:
            window = pd.DataFrame(array_window)
            metrics = window.aggregate(["mean", "std", "min", "max", "median", "kurtosis", "skew"])
            metrics = metrics.transpose()
            metrics["power_energy"] = (window**2).sum()
            """
            Siempre es igual, ahora mismo no sirve de nada
            vc = window.value_counts(normalize=True, sort=False)
            base = e
            metrics["entropy"] = -(vc * np.log(vc)/np.log(base)).sum()
            """
            metrics_windows.append(metrics)
        print("Features creados para", filename)
        np.savez(os.path.join(save_path, filename[:filename.find(".")]+"_features"), features = metrics_windows)