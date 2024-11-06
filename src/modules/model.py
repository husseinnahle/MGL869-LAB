import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
import matplotlib.pyplot as plt
from pickle import dump, load
from simpleNomo import nomogram
import json
import xlsxwriter

version = "2.0.0"

base_dir = Path("/home/augusto/Projects/MGL869-LAB/files")

metrics_path = base_dir / "HiveJavaFiles_2.0.csv"

dataset = pd.read_csv(metrics_path)

# Keep only "Name" and variables columns and "File" rows and
dataset = dataset.query('Kind == "File"').drop("Kind", axis=1)

# Reset dataframe index
dataset = dataset.reset_index(drop=True)

# Read the files with bugs json
with open(base_dir / "files_with_bugs.json", "r") as f:
    files_with_bugs = json.loads(f.read())

# Add "Bugs" column
bugs = pd.DataFrame(np.zeros(len(dataset)), columns=["Bugs"])
dataset = pd.concat([dataset, bugs], axis=1)
java_files = [Path(file).name for file in files_with_bugs[version] if Path(file).suffix == ".java"]
dataset.loc[dataset["Name"].isin(java_files), "Bugs"] = 1
print(f"Total number of .java files: {len(dataset)}")
print(f"Number of .java files in the \"files_with_bugs.json\": {len(java_files)}")
print(f"Number of .java files with bug in the dataset: {len(dataset.loc[dataset["Bugs"] == 1, "Bugs"])}")
print(f"Missing .java files in the dataset:")
for file in java_files:
    if file not in list(dataset["Name"]):
        print(f"    {file}")
print()

# Display initial variable columns
initial_columns = list(dataset.columns[1:-1])
print(f"Initial variable columns: {len(initial_columns)}")
print()

# Drop columns with all NaN
dataset = dataset.dropna(axis=1, how='all')
remaining_columns = list(dataset.columns[1:-1])
print("Drop all NaN columns")
print(f"Remaining columns ({len(remaining_columns)}):")
for column in remaining_columns:
    print(f"    {column}")
dropped_columns = [column for column in initial_columns if column not in remaining_columns]
print(f"Dropped all NaN columns ({len(dropped_columns)}):")
for column in dropped_columns:
    print(f"    {column}")
print()

# Drop columns with all same value
initial_columns = list(dataset.columns[1:-1])
print("Drop same value columns")
print(f"Initial columns: {len(initial_columns)}")
number_unique = dataset.nunique()
cols_to_drop = number_unique[number_unique == 1].index
dataset = dataset.drop(cols_to_drop, axis=1)
remaining_columns = list(dataset.columns[1:-1])
print(f"Remaining columns ({len(remaining_columns)}):")
for column in remaining_columns:
    print(f"    {column}")
dropped_columns = [column for column in initial_columns if column not in remaining_columns]
print(f"Dropped same value columns ({len(dropped_columns)}):")
for column in dropped_columns:
    print(f"    {column}")
print()

# Check for missing values
print("Columns with missing values:")
missing_values = dataset.iloc[:, 1:-1].isnull().sum()
print(missing_values)
print()

# Remove outliers to obtain a better scale in the nomogram
print("Remove outliers:")
print(f"    Initial number of rows in the dataset: {len(dataset)}")
print(f"    Initial number of .java files with bug in the dataset: {len(dataset.loc[dataset["Bugs"] == 1, "Bugs"])}")
dataset_without_outliers = dataset
for column in dataset_without_outliers.columns[1:-1]:
    q_hi = dataset_without_outliers[column].quantile(0.999)
    dataset_without_outliers = dataset_without_outliers[dataset_without_outliers[column] < q_hi]
outliers = dataset[~(dataset.index.isin(list(dataset_without_outliers.index)))]
# Save outliers data to file
outliers.to_csv(base_dir / f"outliers-{version}.csv")
dataset = dataset_without_outliers
print(f"    Final number of rows in the dataset: {len(dataset)}")
print(
    f"    Final number of .java files with bug in the dataset: {len(dataset.loc[dataset["Bugs"] == 1, "Bugs"])}")
print()

# Reset dataframes indexes
dataset = dataset.reset_index(drop=True)
outliers = outliers.reset_index(drop=True)

# Print variables range
print("Variables range:")
for column in dataset.columns[1:-1]:
    print(f"    {column}: {min(dataset[column])} - {max(dataset[column])}")
print()

# Save preprocessed data to file
dataset.to_csv(base_dir / f"metrics_preprocessed-{version}.csv")

# Drop "Name" column
dataset = dataset.drop("Name", axis=1)
outliers = outliers.drop("Name", axis=1)

# Separate data from labels
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Add outliers to test sets
X_outliers = outliers.iloc[:, :-1]
y_outliers = outliers.iloc[:, -1]
X_test = pd.concat([X_test, X_outliers], axis=0)
y_test = pd.concat([y_test, y_outliers], axis=0)

# Generate Logistic Regression classifier
# Optimize the hyperparameters choice with a grid search
param_grid = {
    "penalty": [None, 'l2', 'l1', 'elasticnet'],
    "solver": ['newton-cg', 'newton-cholesky', 'lbfgs', 'sag', 'saga'],
    "max_iter": [100, 300, 500, 1000]
}
# BEST PARAMETERS
# param_grid = {
#     "penalty": ['l2'],
#     "solver": ['lbfgs'],
#     "max_iter": [100]
# }
existing_model = True
try:
    with open(base_dir / f"logistic_regression_model-{version}.pkl", "rb") as f:
        logistic_regression_clf = load(f)
except FileNotFoundError:
    existing_model = False
if not existing_model:
    logistic_regression_clf = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5, scoring='f1', verbose=3)
    logistic_regression_clf.fit(X_train, y_train)
    logistic_regression_clf = LogisticRegression(**logistic_regression_clf.best_params_)
    logistic_regression_clf.fit(X_train, y_train)
    # Save model
    with open(base_dir / f"logistic_regression_model-{version}.pkl", "wb") as f:
        dump(logistic_regression_clf, f, protocol=5)
print(f"logistic_regression_clf best params: {logistic_regression_clf.get_params()}")
print(f"logistic_regression_clf coefficients: {logistic_regression_clf.coef_[0]}")
print(f"logistic_regression_clf intercept_: {logistic_regression_clf.intercept_[0]}")
lr_predicted = logistic_regression_clf.predict(X_test)
lr_predicted_probs = logistic_regression_clf.predict_proba(X_test)[:, 1]
lr_precision, lr_recall, lr_fscore, lr_support = score(y_test, lr_predicted)
print("Logistic Regression classifier performance:")
print(f"precision: {lr_precision}")
print(f"recall: {lr_recall}")
print(f"fscore: {lr_fscore}")
print(f"support: {lr_support}")
print()

# Calculate Logistic Regression AUC
lr_fpr, lr_tpr, lr_thresholds = metrics.roc_curve(y_test, lr_predicted_probs, pos_label=1)
lr_auc = metrics.auc(lr_fpr, lr_tpr)
print(f"Logistic Regression AUC: {lr_auc}")
print()

# Plot the ROC curve (source: https://www.youtube.com/watch?v=VVsvl4WdkfM)
plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, color="blue", label=f"AUC = {lr_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Logistic Regression AUC Curve - version {version}")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(base_dir / f"logistic_regression_auc-{version}.png")

# Generate nomogram configuration file using Logistic Regression coefficients and intercept
workbook = xlsxwriter.Workbook(base_dir / f"nomogram_config-{version}.xlsx")
worksheet = workbook.add_worksheet()
worksheet.write("A1", "feature")
worksheet.write("B1", "coef")
worksheet.write("C1", "min")
worksheet.write("D1", "max")
worksheet.write("E1", "type")
worksheet.write("F1", "position")
worksheet.write("A2", "intercept")
worksheet.write("B2", round(logistic_regression_clf.intercept_[0], 3))
worksheet.write("A3", "threshold")
worksheet.write("B3", 0.5)
# Get variables names from dataframe columns info (remove the index and Bugs columns)
for i, column in enumerate(dataset.columns[:-1], start=0):
    worksheet.write(f"A{i + 4}", column)
    worksheet.write(f"B{i + 4}", round(logistic_regression_clf.coef_[0][i], 3))
    worksheet.write(f"C{i + 4}", min(dataset[column]))
    worksheet.write(f"D{i + 4}", max(dataset[column]))
    worksheet.write(f"E{i + 4}", "continuous")
workbook.close()

# Print nomogram for Logistic Regression
nomogram_fig = nomogram(str(base_dir / f"nomogram_config-{version}.xlsx"), result_title="Bug risk", fig_width=50,
                        single_height=0.45,
                        dpi=300,
                        ax_para={"c": "black", "linewidth": 1.3, "linestyle": "-"},
                        tick_para={"direction": 'in', "length": 3, "width": 1.5, },
                        xtick_para={"fontsize": 10, "fontfamily": "Songti Sc", "fontweight": "bold"},
                        ylabel_para={"fontsize": 12, "fontname": "Songti Sc", "labelpad": 100,
                                     "loc": "center", "color": "black", "rotation": "horizontal"},
                        total_point=100)
nomogram_fig.savefig(base_dir / f"nomogram-{version}.png")

# Generate Random Forest classifier
# Optimize the hyperparameters choice with a grid search
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 4, 8, 16],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2],
    "max_features": ["auto", "sqrt", "log2"],
    "random_state": [0],
}
# BEST PARAMETERS
# param_grid = {
#     "n_estimators": [100],
#     "max_depth": [16],
#     "min_samples_split": [2],
#     "min_samples_leaf": [1],
#     "max_features": ["sqrt"],
#     "random_state": [0],
# }
existing_model = True
try:
    with open(base_dir / f"random_forest_model-{version}.pkl", "rb") as f:
        random_forest_clf = load(f)
except FileNotFoundError:
    existing_model = False
if not existing_model:
    random_forest_clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='f1', verbose=3)
    random_forest_clf.fit(X_train, y_train)
    random_forest_clf = RandomForestClassifier(**random_forest_clf.best_params_)
    random_forest_clf.fit(X_train, y_train)
    # Save model
    with open(base_dir / f"random_forest_model-{version}.pkl", "wb") as f:
        dump(random_forest_clf, f, protocol=5)
print(f"random_forest_clf best params: {random_forest_clf.get_params()}")
rf_predicted = random_forest_clf.predict(X_test)
rf_predicted_probs = random_forest_clf.predict_proba(X_test)[:, 1]
rf_precision, rf_recall, rf_fscore, rf_support = score(y_test, rf_predicted)
print("Random Forest classifier performance:")
print(f"precision: {rf_precision}")
print(f"recall: {rf_recall}")
print(f"fscore: {rf_fscore}")
print(f"support: {rf_support}")
print()

# Calculate Random Forest AUC
rf_fpr, rf_tpr, rf_thresholds = metrics.roc_curve(y_test, rf_predicted_probs, pos_label=1)
rf_auc = metrics.auc(rf_fpr, rf_tpr)
print(f"Random Forest AUC: {rf_auc}")
print()

# Plot the ROC curve (source: https://www.youtube.com/watch?v=VVsvl4WdkfM)
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, color="blue", label=f"AUC = {rf_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Random Forest AUC Curve - version {version}")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(base_dir / f"random_forest_auc-{version}.png")
