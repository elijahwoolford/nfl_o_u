import numpy as np
import pandas as pd
import os
import re

class DataPrep:

    def __init__(self):
        self.data_line_path = "/Users/administrator/Desktop/NFL_O:U_project/nfl_data_lines_2015-2020.csv"
        self.results_path = "/Users/administrator/Desktop/NFL_O:U_project/nfl_results_2015-2020.csv"
        self.cleaned_lines = None
        self.cleaned_results = None
        self.output_path = "/Users/administrator/PycharmProjects/NFL_O_U/output"

    def clean_data_lines(self):
        df = pd.read_csv(self.data_line_path)
        df["Favorite_Home"] = np.where(df["Favorite"].str.contains("at "), 1, 0)
        df["Underdog_Home"] = np.where(df["Favorite_Home"] == 0, 1, 0)
        df["Favorite"] = df["Favorite"].apply(lambda x: x.replace("at ", ""))
        df["Underdog"] = df["Underdog"].apply(lambda x: x.replace("at ", ""))
        df["Week"] = df["Week"].apply(lambda x: int(x))
        self.cleaned_lines = df

        return self.cleaned_lines

    def clean_nfl_results(self):
        print("Cleaning")
        print("-" * 80)
        df = pd.read_csv(self.results_path)
        df = df.dropna(how="all")
        df["Winner/tie"] = df["Winner/tie"].astype(str)
        df["Loser/tie"] = df["Loser/tie"].astype(str)
        df["Winner/tie"] = df["Winner/tie"].apply(lambda x: x.replace(x, x.split(" ")[-1]) if "Washington Football Team" not in x else "Football Team")
        df["Loser/tie"] = df["Loser/tie"].apply(lambda x: str(x).replace(x, x.split(" ")[-1]) if "Washington Football Team" not in x else "Football Team")
        df = df[df["Week"] != "Week"]
        df["Week"] = df["Week"].apply(lambda x: int(x))
        df["year"] = df["year"].apply(lambda x: int(x))
        df["True_total"] = df["PtsW"].astype(int) + df["PtsL"].astype(int)

        self.cleaned_results = df

        return self.cleaned_results

    def combine_data(self, df_data_lines: pd.DataFrame, df_nfl_results: pd.DataFrame):
        dfs = []
        for i in range(2015, 2021):
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
        df_final.to_csv(os.path.join(self.output_path, "combined_results_all.csv"))

        return df_final




if __name__ == '__main__':
    d = DataPrep()
    df_lines = d.clean_data_lines()
    df_results = d.clean_nfl_results()
    d.combine_data(df_lines, df_results)
