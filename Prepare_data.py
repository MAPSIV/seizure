import pandas as pd
import sys
import os

def seizure_id(line):
    F = 64
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

for file in os.listdir(path):
    if file.endswith(".parquet"):
        print("Procesando", file)
        df_data = pd.read_parquet(os.path.join(path, file))
        df_data["id"] = df_data.groupby("filename").cumcount()
        df_data["tag"] = df_data.apply(seizure_id, axis=1)
        df_data.to_parquet(os.path.join(save_path, file))



