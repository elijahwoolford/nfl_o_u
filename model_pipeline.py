import operator

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, ShuffleSplit, \
    RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import feature_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

import various_models

np.random.seed(123)

# nfl_data_lines_path = "/Users/administrator/Desktop/NFL_O:U_project/nfl_data_lines_2017-2019.csv"
# nfl_results_path = "/Users/administrator/Desktop/NFL_O:U_project/nfl_results_2017-2019.csv"
# OUTPUT_DIR = "/Users/administrator/Desktop/NFL_O:U_project"


nfl_data_lines_path = "/Users/administrator/Desktop/NFL_O:U_project/nfl_data_lines_2015-2019.csv"
nfl_results_path = "/Users/administrator/Desktop/NFL_O:U_project/nfl_results_2015-2019.csv"
OUTPUT_DIR = "/Users/administrator/Desktop/NFL_O:U_project"


# Calculate days of rest for each team between each week
# Add defense for prev 3 years
# HPO: shuffle order of features using same random seed
# Features features features!!
# Look into deep learning classification
# save model
# More DATA!!
# Function to get trained model, predict and output predictions
# Look and remove correlated features

def  clean_data_lines():
    df = pd.read_csv(nfl_data_lines_path)
    df["Favorite_Home"] = np.where(df["Favorite"].str.contains("at "), 1, 0)
    df["Underdog_Home"] = np.where(df["Favorite_Home"] == 0, 1, 0)
    df["Favorite"] = df["Favorite"].apply(lambda x: x.replace("at ", ""))
    df["Underdog"] = df["Underdog"].apply(lambda x: x.replace("at ", ""))
    df["Week"] = df["Week"].apply(lambda x: int(x))

    return df


def clean_nfl_results():
    print("Cleaning")
    print("-" * 80)
    df = pd.read_csv(nfl_results_path)
    df = df.dropna(how="all")
    df["Winner/tie"] = df["Winner/tie"].astype(str)
    df["Loser/tie"] = df["Loser/tie"].astype(str)
    df["Winner/tie"] = df["Winner/tie"].apply(lambda x: x.replace(x, x.split(" ")[-1]))
    df["Loser/tie"] = df["Loser/tie"].apply(lambda x: str(x).replace(x, x.split(" ")[-1]))
    df = df[df["Week"] != "Week"]
    df["Week"] = df["Week"].apply(lambda x: int(x))
    df["year"] = df["year"].apply(lambda x: int(x))
    df["True_total"] = df["PtsW"].astype(int) + df["PtsL"].astype(int)

    return df


def combine_data(df_data_lines: pd.DataFrame, df_nfl_results: pd.DataFrame):
    dfs = []
    for i in range(2015, 2020):
        df_year_data_lines = df_data_lines.loc[df_data_lines["Year"] == i]
        df_year_nfl_results = df_nfl_results.loc[df_nfl_results["year"] == i]
        for j in range(17):
            j += 1
            df_curr_results = df_year_nfl_results.loc[df_year_nfl_results["Week"] == j]
            df_curr_lines = df_year_data_lines.loc[df_year_data_lines["Week"] == j]
            df_temp_1 = pd.merge(df_curr_results, df_curr_lines, how='left',
                                 left_on=["Winner/tie", "Loser/tie", "Week"],
                                 right_on=['Favorite', 'Underdog', "Week"])
            df_temp_2 = pd.merge(df_curr_results, df_curr_lines, how='left',
                                 left_on=["Loser/tie", "Winner/tie", "Week"],
                                 right_on=['Favorite', 'Underdog', "Week"])
            df_temp_final = df_temp_1.combine_first(df_temp_2)
            dfs.append(df_temp_final)

    df_final = pd.concat(dfs, sort=True)
    df_final["O/U"] = np.where(df_final["True_total"] >= df_final["Pred_Total"], "Over", "Under")
    df_final.drop(["Unnamed: 5", "Unnamed: 7"], 1)

    return df_final


def plot_sequential_data(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.grid(True)
    plt.savefig("Fitting_analysis.png")


def train_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Cross validaton scores 10 fold: {}".format(cross_val_score(model, X_test, y_test, cv=10)))
    print("-" * 80)

    return model, y_pred


def ensemble(models, X_test, y_test):
    model_pred = []
    y_pred = []
    if isinstance(y_test, pd.Series):
        y_test = y_test.tolist()

    for model in models:
        model_pred.append(model.predict(X_test))

    df_pred = pd.DataFrame()
    for i in enumerate(model_pred):
        df_pred[i[0]] = i[1]

    df_pred_lists = df_pred.values.tolist()
    for list in df_pred_lists:
        final_pred = max(set(list), key=list.count)
        y_pred.append(final_pred)

    print("-" * 80)
    print("Final Classifcation report:\n {}".format(classification_report(y_test, y_pred)))

    tps = get_TPs(y_test, y_pred)
    print("-" * 80)
    print("Total number of true positives = {}".format(tps))
    print("Total number of test samples = {}".format(len(y_test)))

    final_acc = tps / len(y_test)

    print("-" * 80)
    print("Final ensemble true accuracy = {}".format(final_acc))

    return y_test, y_pred, final_acc


def get_TPs(y_test, y_pred):
    tps = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            tps += 1

    return tps


def analyze_fe_importance(importances: list, df_features: pd.DataFrame):
    indices = np.argsort(importances)[::-1]
    names = [df_features.columns[i] for i in indices][:20]

    sns.set(rc={'figure.figsize': (40, 12)})
    ax_pos = sns.barplot(x=names, y=importances[indices][:20])
    ax_pos.set(xlabel="Feature Importances")
    fig_ax_pos = ax_pos.get_figure()
    fig_ax_pos.savefig("Feature_Importantances" + ".png")


def analyze_results(y_true, y_pred, df_data_lines, df_final):
    # Maybe stats?

    df_data_lines_test = df_data_lines.tail(len(y_true)).drop(["Favorite_Home", "Underdog_Home"], axis=1).reset_index(
        drop=True)
    df_final_test = df_final.tail(len(y_true))
    df_final_test = df_final_test[["PtsW", "PtsL", "True_total"]].reset_index(drop=True)

    df_analysis = pd.DataFrame()
    df_analysis["True_Results"] = y_true
    df_analysis["Pred_Results"] = y_pred
    df_analysis["Correct"] = np.where(df_analysis["True_Results"] == df_analysis["Pred_Results"], "True", "False")
    df_analysis_final = pd.concat([df_data_lines_test, df_analysis], axis=1)
    df_analysis_final = pd.concat([df_analysis_final, df_final_test], axis=1)
    df_analysis_final.to_csv("model_analysis.csv")

    return df_analysis_final


def get_best_rf_model(X_train, y_train):
    parameter_space = {'criterion': ['gini', 'entropy'],
                       'bootstrap': [True, False],
                       'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                       'max_features': ['auto', 'sqrt'],
                       'min_samples_leaf': [1, 2, 4],
                       'min_samples_split': [2, 5, 10],
                       'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

    rf_model = RandomForestClassifier(random_state=111)

    clf = RandomizedSearchCV(rf_model, parameter_space, n_jobs=-1, cv=5, n_iter=50, random_state=42)
    clf.fit(X_train, y_train)

    return clf


def random_forest_model(X_train, y_train, X_test, y_test, df_features, random_state):
    rf_model = RandomForestClassifier(random_state=random_state)
    rf_model_train, y_pred = train_model(X_train, X_test, y_train, y_test, rf_model)
    importances = rf_model_train.feature_importances_[:20]
    analyze_fe_importance(importances, df_features)

    return rf_model


def pipeline():
    # results_encoded = os.path.join(output_file, "encoded_results.csv")
    combined_results = os.path.join(OUTPUT_DIR, "combined_results_all.csv")
    if os.path.exists(combined_results):
        df_final = pd.read_csv(combined_results)
        df_data_lines = clean_data_lines()
    else:
        df_nfl_results = clean_nfl_results()
        df_data_lines = clean_data_lines()
        df_final = combine_data(df_data_lines, df_nfl_results)
        df_final.to_csv(combined_results)

    X_train, X_test, y_train, y_test = feature_selection.create_features(df_final)
    df_features = pd.read_csv(os.path.join(OUTPUT_DIR, "df_features.csv"))

    # TRAINING
    print("Training Models\n")
    models = []

    # lr_model = LogisticRegression(random_state=111, C=100)
    # lr_model_train = train_model(X_train, X_test, y_train, y_test, lr_model)
    # models.append(lr_model_train)

    # rf_model_train = get_best_rf_model(X_train, y_train)
    # models.append(rf_model_train)

    # lin_svc_model = LinearSVC()
    # lin_svc_model_train, y_pred = train_model(X_train, X_test, y_train, y_test, lin_svc_model)
    # models.append(lin_svc_model_train)

    # rex_model = ExtraTreesClassifier()
    # rex_model_train, y_pred = train_model(X_train, X_test, y_train, y_test, rex_model)
    # models.append(rex_model_train)
    #
    # nb_model = GaussianNB()
    # nb_model_train, y_pred = train_model(X_train, X_test, y_train, y_test, nb_model)
    # models.append(nb_model_train)

    rf_model = random_forest_model(X_train, y_train, X_test, y_test, df_features, 1111)
    models.append(rf_model)

    # dl_model = various_models.deep_learning_model(X_train, y_train, X_test, y_test)

    y_test, y_pred, final_accs = ensemble(models, X_test, y_test)
    analyze_results(y_test, y_pred, df_data_lines, df_final)


if __name__ == '__main__':
    pipeline()
