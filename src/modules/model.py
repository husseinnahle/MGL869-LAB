import os
import logging
import math
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
from collections import Counter
import matplotlib.pyplot as plt
from pickle import dump, load
from simpleNomo import nomogram
import json
import xlsxwriter
from statistics import stdev

version = "2_2_0"

dots_separated_version = ".".join(version.split("_"))

base_dir = Path(os.path.realpath(__file__)).parent.parent.parent / "data" / "metrics"

logging.basicConfig(filename=base_dir / f"logs_{version}.log",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.getLogger('matplotlib').setLevel(logging.ERROR)

all_metrics_path = base_dir / f"und_hive_all_metrics_{version}.csv"

if not all_metrics_path.exists():

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
        else:  # Mean
            if method_data[column_name].empty:
                return np.nan
            return method_data[column_name].mean()


    classes_metrics = ["CountClassBase", "CountClassCoupled", "CountClassDerived", "MaxInheritanceTree",
                       "PercentLackOfCohesion"]
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
                filtered_dataset.loc[filtered_dataset["Name"] == file_name, col + spec] = 0 if np.isnan(
                    value) else value

    # Remove the specified columns
    filtered_dataset = filtered_dataset.drop(columns=methods_metrics)

    filtered_dataset.to_csv(base_dir / f"und_hive_all_metrics_{version}.csv", index=False)
else:
    filtered_dataset = pd.read_csv(all_metrics_path)

# Keep only "Name" and variables columns and "File" rows and
#dataset = dataset.query('Kind == "File"').drop("Kind", axis=1)

# Reset dataframe index
#filtered_dataset = filtered_dataset.reset_index(drop=True)

# Read the files with bugs json
with open(base_dir.parent / f"files_with_bugs.json", "r") as f:
    files_with_bugs = json.loads(f.read())

# Add "Bugs" column
bugs = pd.DataFrame(np.zeros(len(filtered_dataset)), columns=["Bugs"])
filtered_dataset = pd.concat([filtered_dataset, bugs], axis=1)
java_files = [Path(file).name for file in files_with_bugs[dots_separated_version] if Path(file).suffix == ".java"]
filtered_dataset.loc[filtered_dataset["Name"].isin(java_files), "Bugs"] = 1
logging.info(f"Total number of .java files: {len(filtered_dataset)}")
logging.info(f"Number of .java files in the \"files_with_bugs_{version}.json\": {len(java_files)}")
logging.info(f"Number of .java files with bug in the filtered_dataset: {len(filtered_dataset.loc[filtered_dataset["Bugs"] == 1, "Bugs"])}")
logging.info(f"Missing .java files in the filtered_dataset:")
for file in java_files:
    if file not in list(filtered_dataset["Name"]):
        logging.info(f"    {file}")
logging.info("")

# Display initial variable columns
initial_columns = list(filtered_dataset.columns[1:-1])
logging.info(f"Initial variable columns: {len(initial_columns)}")
logging.info("")

# Drop columns with all NaN
filtered_dataset = filtered_dataset.dropna(axis=1, how='all')
remaining_columns = list(filtered_dataset.columns[1:-1])
logging.info("Drop all NaN columns")
logging.info(f"Remaining columns ({len(remaining_columns)}):")
for column in remaining_columns:
    logging.info(f"    {column}")
dropped_columns = [column for column in initial_columns if column not in remaining_columns]
logging.info(f"Dropped all NaN columns ({len(dropped_columns)}):")
for column in dropped_columns:
    logging.info(f"    {column}")
logging.info("")

# Drop columns with all same value
initial_columns = list(filtered_dataset.columns[1:-1])
logging.info("Drop same value columns")
logging.info(f"Initial columns: {len(initial_columns)}")
number_unique = filtered_dataset.nunique()
cols_to_drop = number_unique[number_unique == 1].index
filtered_dataset = filtered_dataset.drop(cols_to_drop, axis=1)
remaining_columns = list(filtered_dataset.columns[1:-1])
logging.info(f"Remaining columns ({len(remaining_columns)}):")
for column in remaining_columns:
    logging.info(f"    {column}")
dropped_columns = [column for column in initial_columns if column not in remaining_columns]
logging.info(f"Dropped same value columns ({len(dropped_columns)}):")
for column in dropped_columns:
    logging.info(f"    {column}")
logging.info("")

# Check for missing values
logging.info("Columns with missing values:")
missing_values = filtered_dataset.iloc[:, 1:-1].isnull().sum()
for column in missing_values.index:
    logging.info(f"    {column}: {missing_values[column]}")
logging.info("")


# Checking boxplots (ref.: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way)
def boxplots_custom(filtered_dataset, columns_list, rows, cols, suptitle):
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(13, 50))
    fig.suptitle(suptitle, y=1, size=25)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        sns.boxplot(data=filtered_dataset[data], orient='h', ax=axs[i])
        axs[i].set_title(data + ', skewness is: ' + str(round(filtered_dataset[data].skew(axis=0, skipna=True), 2)))


columns_list = list(filtered_dataset.columns[1:-1])
boxplots_custom(filtered_dataset=filtered_dataset, columns_list=columns_list, rows=math.ceil(len(columns_list) / 3), cols=3,
                suptitle='Boxplots for each variable')
plt.tight_layout()
plt.savefig(base_dir / f"boxplots_{version}.png")


def IQR_method(df, n, features):
    """
    Takes a dataframe and returns an index list corresponding to the observations
    containing more than n outliers according to the Tukey IQR method.
    Ref.: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way
    """
    outlier_list = []

    for column in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        # Determining a list of indices of outliers
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
        # appending the list of outliers
        outlier_list.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

    return multiple_outliers


# Remove outliers (save the outliers to disk)
# Use 10 as the `n` argument of `IQR_method` to allow more outliers to be kept, otherwise most of the files with bugs
# where being removed
logging.info("Remove outliers:")
logging.info(f"    Initial number of rows in the filtered_dataset: {len(filtered_dataset)}")
logging.info(
    f"    Initial number of .java files with bug in the filtered_dataset: {len(filtered_dataset.loc[filtered_dataset["Bugs"] == 1, "Bugs"])}")
outliers_IQR = IQR_method(filtered_dataset, 10, columns_list)
outliers = filtered_dataset.loc[outliers_IQR].reset_index(drop=True)
logging.info(f"    Total number of outliers is: {len(outliers_IQR)}")
outliers.to_csv(base_dir / f"outliers_{version}.csv", index=False)
filtered_dataset = filtered_dataset.drop(outliers_IQR, axis=0).reset_index(drop=True)
logging.info(f"    Final number of rows in the filtered_dataset: {len(filtered_dataset)}")
logging.info(
    f"    Final number of .java files with bug in the filtered_dataset: {len(filtered_dataset.loc[filtered_dataset["Bugs"] == 1, "Bugs"])}")
logging.info("")

# Remove correlated columns
# Ref.: https://www.kaggle.com/code/prashant111/comprehensive-guide-on-feature-selection#2.6-Correlation-Matrix-with-Heatmap-
corr_matrix = filtered_dataset.iloc[:, 1:-1].corr()

# Create correlation heatmap
plt.figure(figsize=(77,75))
plt.title(f'Correlation Heatmap version {dots_separated_version}')
a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
a.set_xticklabels(a.get_xticklabels(), rotation=30)
a.set_yticklabels(a.get_yticklabels(), rotation=30)
plt.savefig(base_dir / f"correlation_heatmap_{version}.png")

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.9
to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.9)]
logging.info("Correlated columns to be dropped:")
for column in to_drop:
    correlated_to = list(upper[upper[column].abs() > 0.9].index)
    logging.info(f"    {column}, correlated to: {correlated_to}")

# Drop correlated columns
filtered_dataset = filtered_dataset.drop(to_drop, axis=1)
outliers = outliers.drop(to_drop, axis=1)

# Print variables range
logging.info("Variables range:")
for column in filtered_dataset.columns[1:-1]:
    logging.info(f"    {column}: {min(filtered_dataset[column])} - {max(filtered_dataset[column])}")
logging.info("")

# Save preprocessed data to file
filtered_dataset.to_csv(base_dir / f"und_hive_metrics_preprocessed_{version}.csv", index=False)

# Drop "Name" column
filtered_dataset = filtered_dataset.drop("Name", axis=1)
outliers = outliers.drop("Name", axis=1)

# Separate data from labels
X = filtered_dataset.iloc[:, :-1]
y = filtered_dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Add outliers to test sets
X_outliers = outliers.iloc[:, :-1]
y_outliers = outliers.iloc[:, -1]
X_test = pd.concat([X_test, X_outliers], axis=0)
y_test = pd.concat([y_test, y_outliers], axis=0)

# Set 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=False)

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
    with open(base_dir / f"logistic_regression_model_{version}.pkl", "rb") as f:
        logistic_regression_clf = load(f)
except FileNotFoundError:
    existing_model = False
if not existing_model:
    logistic_regression_grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=kf, scoring='precision',
                                            verbose=3)
    logistic_regression_grid.fit(X_train, y_train)
    logistic_regression_clf = logistic_regression_grid.best_estimator_
    # Save model
    with open(base_dir / f"logistic_regression_model_{version}.pkl", "wb") as f:
        dump(logistic_regression_clf, f, protocol=5)
logging.info(f"logistic_regression_clf best params: {logistic_regression_clf.get_params()}")
logging.info(f"logistic_regression_clf coefficients: {logistic_regression_clf.coef_[0]}")
logging.info(f"logistic_regression_clf intercept_: {logistic_regression_clf.intercept_[0]}")

# Calculate 10-fold cross validation scores
# Ref.: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way
score_lr = cross_val_score(logistic_regression_clf, X_train, y_train, cv=kf, scoring='precision')
lr_score = score_lr.mean()
lr_stdev = stdev(score_lr)
logging.info(f'Logistic Regression Cross Validation Precision scores are: {score_lr}')
logging.info(f'Logistic Regression Average Cross Validation Precision score: {lr_score}')
logging.info(f'Logistic Regression Cross Validation Precision standard deviation: {lr_stdev}')
lr_predicted = logistic_regression_clf.predict(X_test)
lr_predicted_probs = logistic_regression_clf.predict_proba(X_test)[:, 1]
lr_precision, lr_recall, lr_fscore, lr_support = score(y_test, lr_predicted)
logging.info("Logistic Regression classifier performance:")
logging.info(f"precision: {lr_precision}")
logging.info(f"recall: {lr_recall}")
logging.info(f"fscore: {lr_fscore}")
logging.info(f"support: {lr_support}")
logging.info("")

# Calculate Logistic Regression AUC
lr_fpr, lr_tpr, lr_thresholds = metrics.roc_curve(y_test, lr_predicted_probs, pos_label=1)
lr_auc = metrics.auc(lr_fpr, lr_tpr)
logging.info(f"Logistic Regression AUC: {lr_auc}")
logging.info("")

# Plot the ROC curve (source: https://www.youtube.com/watch?v=VVsvl4WdkfM)
plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, color="blue", label=f"AUC = {lr_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Logistic Regression AUC Curve - version {dots_separated_version}")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(base_dir / f"logistic_regression_auc_{version}.png")

# Generate nomogram configuration file using Logistic Regression coefficients and intercept
workbook = xlsxwriter.Workbook(base_dir / f"nomogram_config_{version}.xlsx")
worksheet = workbook.add_worksheet()
worksheet.write("A1", "feature")
worksheet.write("B1", "coef")
worksheet.write("C1", "min")
worksheet.write("D1", "max")
worksheet.write("E1", "type")
worksheet.write("F1", "position")
worksheet.write("A2", "intercept")
worksheet.write("B2", round(logistic_regression_clf.intercept_[0], 4))
worksheet.write("A3", "threshold")
worksheet.write("B3", 0.5)
# Get variables names from dataframe columns info (remove the index and Bugs columns)
for i, column in enumerate(filtered_dataset.columns[:-1], start=0):
    worksheet.write(f"A{i + 4}", column)
    worksheet.write(f"B{i + 4}", round(logistic_regression_clf.coef_[0][i], 4))
    worksheet.write(f"C{i + 4}", min(filtered_dataset[column]))
    worksheet.write(f"D{i + 4}", max(filtered_dataset[column]))
    worksheet.write(f"E{i + 4}", "continuous")
workbook.close()

# Print nomogram for Logistic Regression
nomogram_fig = nomogram(str(base_dir / f"nomogram_config_{version}.xlsx"), result_title="Bug risk", fig_width=50,
                        single_height=0.45,
                        dpi=300,
                        ax_para={"c": "black", "linewidth": 1.3, "linestyle": "-"},
                        tick_para={"direction": 'in', "length": 3, "width": 1.5, },
                        xtick_para={"fontsize": 10, "fontfamily": "Songti Sc", "fontweight": "bold"},
                        ylabel_para={"fontsize": 12, "fontname": "Songti Sc", "labelpad": 100,
                                     "loc": "center", "color": "black", "rotation": "horizontal"},
                        total_point=100)
nomogram_fig.savefig(base_dir / f"nomogram_{version}.png")

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
    with open(base_dir / f"random_forest_model_{version}.pkl", "rb") as f:
        random_forest_clf = load(f)
except FileNotFoundError:
    existing_model = False
if not existing_model:
    random_forest_grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=kf, scoring='precision',
                                      verbose=3)
    random_forest_grid.fit(X_train, y_train)
    random_forest_clf = random_forest_grid.best_estimator_
    # Save model
    with open(base_dir / f"random_forest_model_{version}.pkl", "wb") as f:
        dump(random_forest_clf, f, protocol=5)
logging.info(f"random_forest_clf best params: {random_forest_clf.get_params()}")

# Calculate 10-fold cross validation scores
# Ref.: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way
score_rf = cross_val_score(random_forest_clf, X_train, y_train, cv=kf, scoring='precision')
rf_score = score_rf.mean()
rf_stdev = stdev(score_rf)
logging.info(f'Randon Forest Cross Validation Precision scores are: {score_rf}')
logging.info(f'Randon Forest Average Cross Validation Precision score: {rf_score}')
logging.info(f'Randon Forest Cross Validation Precision standard deviation: {rf_stdev}')
rf_predicted = random_forest_clf.predict(X_test)
rf_predicted_probs = random_forest_clf.predict_proba(X_test)[:, 1]
rf_precision, rf_recall, rf_fscore, rf_support = score(y_test, rf_predicted)
logging.info("Random Forest classifier performance:")
logging.info(f"precision: {rf_precision}")
logging.info(f"recall: {rf_recall}")
logging.info(f"fscore: {rf_fscore}")
logging.info(f"support: {rf_support}")
logging.info("")

# Calculate Random Forest AUC
rf_fpr, rf_tpr, rf_thresholds = metrics.roc_curve(y_test, rf_predicted_probs, pos_label=1)
rf_auc = metrics.auc(rf_fpr, rf_tpr)
logging.info(f"Random Forest AUC: {rf_auc}")
logging.info("")

# Plot the ROC curve (source: https://www.youtube.com/watch?v=VVsvl4WdkfM)
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, color="blue", label=f"AUC = {rf_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Random Forest AUC Curve - version {dots_separated_version}")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(base_dir / f"random_forest_auc_{version}.png")
