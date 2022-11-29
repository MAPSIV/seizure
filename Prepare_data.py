import pandas as pd
import sys
import os
import numpy as np

def seizure_id(line):
    S = 30
    if line["type"] == "normal":
        return 0
    for a in df_annotations[df_annotations["filename"] == line["filename"]].values:
        if a[5]*F<=line["id"] and a[6]*F>=line["id"]: # Seizure
            return 1
        if a[5]*F > line["id"] and a[5]*F - F*S <= line["id"]: # Antes Seizure
            return 2
        if a[6]*F  + F*S >= line["id"] and a[6]*F < line["id"]: # Despues Seizure
            return 3
    return 0

path  = sys.argv[1]
df_annotations = pd.read_excel(os.path.join(path, "df_annotation_full.xlsx"))

save_path = "Datos_procesados"
if not os.path.exists('Datos_procesados'):
   os.makedirs('Datos_procesados')

if len(sys.argv)>2:
    F = int(sys.argv[2])
else:
    F = 128
Division_window = F//2

for filename in os.listdir(path):
    if filename.endswith(".parquet"):
        print("Procesando", filename)
        df_data = pd.read_parquet(os.path.join(path, filename))
        df_data["id"] = df_data.groupby("filename").cumcount()
        df_data["tag"] = df_data.apply(seizure_id, axis=1)
        df_data.to_parquet(os.path.join(save_path, filename))
        windows = []
        metadatos = []
        for file in df_data["filename"].unique():
            df_file = df_data[df_data["filename"] == file]
            for tag in [0, 1, 2, 3]:
                df_tag = df_file[df_file["tag"] == tag].drop(["type", "PatID", "filename", "id", "tag"], axis=1)
                df_tag = df_tag.drop(df_tag.tail(df_tag.shape[0]%F).index)
                if df_tag.shape[0]>0:
                    windows = windows + np.array_split(df_tag, df_tag.shape[0]/F)
                    metadatos = metadatos + [[file, tag] for _ in range(int(df_tag.shape[0]/F))]
                    if tag == 1:
                        for i in range(Division_window):
                            df_tag = np.roll(df_tag, int(-F/Division_window), axis=0)
                            windows = windows + np.array_split(df_tag, df_tag.shape[0]/F)
                            metadatos = metadatos + [[file, tag] for _ in range(int(df_tag.shape[0]/F))]
        np.savez(os.path.join(save_path, filename[:filename.find(".")]+"_windows"), windows = windows)
        metadatos = pd.DataFrame(metadatos, columns=["filename", "tag"])
        metadatos.to_parquet(os.path.join(save_path, filename[:filename.find(".")]+"_metadatos.parquet"))
