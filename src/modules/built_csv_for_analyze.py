import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Enregistrez l'heure de début
start_time = time.time()

version = "3_1_0"

base_dir = Path(os.path.realpath(__file__)).parent.parent.parent / "data" / "metrics"

logging.basicConfig(filename=base_dir / f"logs_{version}.log",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.getLogger('matplotlib').setLevel(logging.ERROR)

#all_metrics_path = base_dir / f"und_hive_all_metrics_{version}.csv"
metrics_path = base_dir / f"und_hive_{version}.csv"

dataset = pd.read_csv(metrics_path)
# List of column names to be removed
columns_to_remove = ["CCViolDensityCode", "CCViolDensityLine", "CountCCViol", "CountCCViolType",
                     "CountClassCoupledModified", "CountDeclExecutableUnit", "CountDeclFile",
                     "CountDeclMethodAll", "Cyclomatic", "PercentLackOfCohesionModified"]
# Remove the specified columns
dataset = dataset.drop(columns=columns_to_remove)

filtered_dataset = dataset[dataset["Kind"] == "File"].copy()
dataset = dataset[dataset["Kind"] != "File"]

def calculate_value_class(dataset, column_name, mask):
    class_data = dataset.loc[mask & dataset["Kind"].str.contains("Class")]
    if class_data.empty:
        return np.nan

    if class_data[column_name].empty:
        return np.nan
    return class_data[column_name].mean()

def calculate_value_method(dataset, column_name, specification, mask):
    method_data = dataset.loc[mask & dataset["Kind"].str.contains("Method")]
    if method_data.empty:
        return np.nan

    if "Min" in specification:
        return method_data[column_name].min()
    elif "Max" in specification:
        return method_data[column_name].max()
    else: #Mean
        if method_data[column_name].empty:
            return np.nan
        return method_data[column_name].mean()

classes_metrics = ["CountClassBase", "CountClassCoupled", "CountClassDerived", "MaxInheritanceTree", "PercentLackOfCohesion"]
methods_metrics = ["CountInput", "CountOutput", "CountPath", "MaxNesting"]
methods_specification = ["Min", "Mean", "Max"]

file_names = filtered_dataset["Name"].apply(lambda x: Path(x).stem)
masks = {file_name: dataset["Name"].str.contains(file_name) for file_name in file_names}

for i, file_name in enumerate(filtered_dataset["Name"], start=1):
    logging.info(f"{i} - {file_name}")
    file_name_without_extension = Path(file_name).stem

    mask = masks[file_name_without_extension]
    if not mask.any():
        logging.info(f"{file_name} not found in dataset, skipping...")
        continue

    class_values = {col: calculate_value_class(dataset, col, mask) for col in classes_metrics}
    for col, value in class_values.items():
        filtered_dataset.loc[filtered_dataset["Name"] == file_name, col] = 0 if np.isnan(value) else value

    for col in methods_metrics:
        method_values = {spec: calculate_value_method(dataset, col, spec, mask) for spec in methods_specification}
        for spec, value in method_values.items():
            filtered_dataset.loc[filtered_dataset["Name"] == file_name, col + spec] = 0 if np.isnan(value) else value



# Remove the specified columns
filtered_dataset = filtered_dataset.drop(columns=methods_metrics)

filtered_dataset.to_csv(base_dir / f"und_hive_all_metrics_{version}.csv", index=False)

# Enregistrez l'heure de fin
end_time = time.time()

# Calculez le temps d'exécution
execution_time = end_time - start_time
print(f"Le temps d'exécution est de {execution_time} secondes")