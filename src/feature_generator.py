import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, normalize, MinMaxScaler


class FeatureGenerator:

    # Todo: Add current number of wins and losses

    def __init__(self):
        self.gold_results = pd.read_csv("/Users/administrator/PycharmProjects/NFL_O_U/output/combined_results_all.csv").dropna(how="all")
        self.defense_results = pd.read_csv("/Users/administrator/Desktop/NFL_O:U_project/2015_2019_def_stats.csv")
        self.raw_features = self.gold_results[
            ["Day", "Spread", "Pred_Total", "Favorite", "Underdog", "Away Money Line", "Home Money Line", "Year",
             "Week", "O/U"]]
        self.raw_defense_features = None
        self.total_features = None
        self.output_path = "/Users/administrator/PycharmProjects/NFL_O_U/output"

    def generator(self):
        self.clean_features_defense()
        self.total_features = self.get_features(self.raw_features, self.raw_defense_features)

        return self.total_features

    def clean_features_defense(self):
        self.raw_defense_features = self.defense_results.dropna()
        self.raw_defense_features["name_abb"] = self.raw_defense_features["Name"].apply(
            lambda x: str(x).split(" ")[-1])
        self.raw_defense_features["prev_week"] = self.raw_defense_features["Week"].apply(lambda x: int(x) - 1)

    @staticmethod
    def get_features(df_features, df_def_stats):
        n = df_features.shape[0]
        df_features = df_features.reset_index()
        def_feature_columns = ["TE allowed", "RB allowed", "K allowed", "QB allowed", "Total FPts allowed"]
        for feature in def_feature_columns:
            prev_favorite = []
            prev_underdog = []
            for i in range(n):
                curr_week = df_features.loc[i, "Week"]
                curr_fav = df_features.loc[i, "Favorite"]
                curr_und = df_features.loc[i, "Underdog"]
                df_year_def_stats = df_def_stats[df_def_stats["Year"] == df_features.loc[i, "Year"]]
                df_week_def_stats = df_year_def_stats[df_year_def_stats["Week"] == curr_week - 1]
                if curr_week != 1:
                    try:
                        curr_fav_prev_stats = \
                            df_week_def_stats.loc[df_week_def_stats["name_abb"] == curr_fav][
                                feature].tolist()[0]
                    except IndexError:
                        curr_fav_prev_stats = df_def_stats[feature].mean()
                    try:
                        curr_underdog_prev_stats = \
                            df_week_def_stats.loc[df_week_def_stats["name_abb"] == curr_und][
                                feature].tolist()[0]
                    except IndexError:
                        curr_underdog_prev_stats = df_def_stats[feature].mean()
                    prev_favorite.append(curr_fav_prev_stats)
                    prev_underdog.append(curr_underdog_prev_stats)
                else:
                    avg_year_fav = df_def_stats[df_def_stats["name_abb"] == curr_fav][feature].mean()
                    avg_year_und = df_def_stats[df_def_stats["name_abb"] == curr_und][feature].mean()
                    prev_favorite.append(avg_year_fav)
                    prev_underdog.append(avg_year_und)

            df_features["prev_" + feature + "_Favorite"] = prev_favorite
            df_features["prev_" + feature + "_Underdog"] = prev_underdog

        return df_features

    def clean_features(self):
        self.total_features = self.total_features.drop(["Favorite", "Underdog", "index"], 1)
        enc = OrdinalEncoder()
        ord_cols = ["Year", "Day", "Week"]
        self.total_features[ord_cols] = enc.fit_transform(self.total_features[ord_cols])
        for col in self.total_features.columns:
            if col not in ord_cols and col != "O/U":
                self.total_features[col] = normalize(np.array(self.total_features[col]).reshape(-1, 1), axis=0)

        # self.total_features = pd.get_dummies(self.total_features, columns=["Year", "Day", "Week"])

    def save(self):
        self.total_features.to_csv(os.path.join(self.output_path, "total_features_2020.csv"), index=False)


if __name__ == '__main__':
    f = FeatureGenerator()
    f.generator()
    f.clean_features()
    f.save()
