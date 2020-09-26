import os

import featuretools as ft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

OUTPUT_DIR = "/Users/administrator/Desktop/NFL_O:U_project"
defense_path = "/Users/administrator/Desktop/NFL_O:U_project/2015_2019_def_stats.csv"


def entity_features(df_final: pd.DataFrame):
    # 'add_numeric'
    es = ft.EntitySet(id='Feature Extension')
    es.entity_from_dataframe(entity_id='hr', dataframe=df_final, index='index')

    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='hr',
                                          trans_primitives=['multiply_numeric'],
                                          verbose=True)

    return feature_matrix, feature_defs


def create_features(df_final: pd.DataFrame):
    # "Home Money Line"
    # "Favorite_Home"
    # "Underdog_Home"

    df_features = df_final[
        ["Day", "Favorite", "Spread", "Underdog", "Pred_Total", "Away Money Line", "Year", "Week",
         "O/U"]]

    df_def_stats = clean_features_defense(defense_path)

    df_features = get_features_defense(df_features, df_def_stats).drop("Week", 1)

    # df_features["Matchup"] = df_features["Favorite"] + "_" + df_features["Underdog"]

    df_features_encoded = pd.get_dummies(df_features, columns=["Favorite", "Underdog", "Year"])
    df_features_encoded["Day"] = np.where(df_features["Day"] == "Thu", 1, 0)

    # Feature playground:

    df_features_encoded["AMLxDay"] = df_features_encoded["Away Money Line"] * df_features_encoded["Day"]

    df_features_encoded = df_features_encoded.reset_index(drop=True).dropna(axis=0)

    ######
    df_features_num = df_features_encoded[
        ["Day", "Spread", "Pred_Total", "Away Money Line", "prev_TE allowed_Loser"]]

    # df_features_num_encoded = scaler(df_features_num)

    feature_matrix, feature_defs = entity_features(df_features_num)

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(feature_matrix.corr(), vmin=-1, vmax=1, cmap="RdBu", linewidths=1, ax=ax, annot=True)
    # sns.heatmap(df_features_num.corr(), vmin=-1, vmax=1, cmap="RdBu", linewidths=1, ax=ax, annot=True)
    fig.savefig("heatmap.png")

    # Train test split:

    df_X = df_features_encoded.drop("O/U", axis=1)

    df_X.to_csv(os.path.join(OUTPUT_DIR, "df_features.csv"))
    # X = scalar.fit_transform(df_X)
    X = df_features_encoded.drop("O/U", axis=1)
    y = df_features_encoded["O/U"]

    print("Total Number of Over: {}".format((y == "Over").sum()))
    print("Total Number of Under: {}".format((y == "Under").sum()))
    print("-" * 80)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Total Number of Training Over: {}".format((y_train == "Over").sum()))
    print("Total Number of Training Under: {}".format((y_train == "Under").sum()))
    print("-" * 80)

    print("Total Number of Test Over: {}".format((y == "Over").sum() - (y_train == "Over").sum()))
    print("Total Number of Test Under: {}".format((y == "Under").sum() - (y_train == "Under").sum()))
    print("-" * 80)

    return X_train, X_test, y_train, y_test


def clean_features_defense(def_stats_path: str):
    df_def_stats = pd.read_csv(def_stats_path)
    df_def_stats = df_def_stats.dropna()
    df_def_stats["name_abb"] = df_def_stats["Name"].apply(lambda x: str(x).split(" ")[-1])
    df_def_stats["prev_week"] = df_def_stats["Week"].apply(lambda x: int(x) - 1)

    return df_def_stats


def get_features_defense(df_features, df_def_stats):
    n = df_features.shape[0]
    df_features = df_features.reset_index()
    feature_columns = ["TE allowed", "RB allowed", "K allowed", "QB allowed"]
    # feature_columns = ["TE allowed", "Total FPts allowed"]
    for feature in feature_columns:
        prev_winner = []
        prev_loser = []
        for i in range(n):
            curr_winner = df_features.loc[i, "Favorite"]
            curr_loser = df_features.loc[i, "Underdog"]
            curr_week = df_features.loc[i, "Week"]
            curr_year = df_features.loc[i, "Year"]
            df_year_def_stats = df_def_stats[df_def_stats["Year"] == curr_year]
            df_week_def_stats = df_year_def_stats[df_year_def_stats["Week"] == curr_week - 1]
            if curr_week != 1:
                try:
                    curr_winner_prev = \
                        df_week_def_stats.loc[df_week_def_stats["name_abb"] == curr_winner][feature].tolist()[0]
                except IndexError:
                    curr_winner_prev = df_def_stats[feature].mean()
                try:
                    curr_loser_prev = \
                        df_week_def_stats.loc[df_week_def_stats["name_abb"] == curr_loser][feature].tolist()[0]
                except IndexError:
                    curr_loser_prev = df_def_stats[feature].mean()

                prev_winner.append(curr_winner_prev)
                prev_loser.append(curr_loser_prev)
            else:
                avg_year = df_def_stats[feature].mean()
                prev_winner.append(avg_year)
                prev_loser.append(avg_year)

        df_features["prev_" + feature + "_Winner"] = prev_winner
        df_features["prev_" + feature + "_Loser"] = prev_loser

    return df_features


def scaler(df: pd.DataFrame):
    # PREPROCESSING
    # 0-1
    # scalar = preprocessing.StandardScaler()
    min_max_neg = MinMaxScaler(feature_range=(-1, 1))
    min_max_pos = MinMaxScaler(feature_range=(0, 1))
    # df["Away Money Line"] = min_max_neg.fit_transform(df["Away Money Line"].values.reshape(-1, 1))
    # df["Spread"] = min_max_pos.fit_transform(df["Spread"].values.reshape(-1, 1))
    # df["Pred_Total"] = min_max_pos.fit_transform(df["Pred_Total"].values.reshape(-1, 1))

    return df

    # LEARNING CURVE
    # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    # title = "Learning Curves Random Fores"
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    # model = RandomForestClassifier(random_state=111)
    # plot_learning_curve(model, title, X_train, y_train, axes=axes[:, 0], ylim=(0.0, 1.01), cv=cv)
    #
    # title = "Learning Curves Linear SVC"
    # model = LinearSVC()
    # plot_learning_curve(model, title, X_train, y_train, axes=axes[:, 1], ylim=(0.0, 1.01), cv=cv)
    #
    # plt.savefig("Learning_curve_plots.png")

if __name__ == '__main__':
    df_def_stats = clean_features_defense(defense_path)
    df_final = pd.read_csv("/Users/administrator/Desktop/tester/combined_results_all.csv")
    df_features = df_final[["Day", "Favorite", "Spread", "Underdog", "Pred_Total", "Away Money Line",
                            "Home Money Line", "Favorite_Home", "Underdog_Home", "Year", "Week", "O/U"]]
    get_features_defense(df_features, df_def_stats)
