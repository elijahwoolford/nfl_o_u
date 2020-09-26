import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate

warnings.simplefilter(action='ignore', category=FutureWarning)


class Model:

    def __init__(self, df_features: pd.DataFrame):
        self.model = None
        # self.clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        self.clf = RandomForestClassifier(n_estimators=500, random_state=42)
        self.features = df_features
        self.pred = None
        self.gold = None
        self.output_path = "/Users/administrator/PycharmProjects/NFL_O_U/output"
        self.model_file_name = "rf_ou_v1.0"

    def split_data(self):
        X = self.features.drop(columns=["O/U"])
        y = np.where(self.features["O/U"] == "Over", 1, 0)
        # scaler = MinMaxScaler()
        # scaler.fit(X)
        # # X = scaler.transform(X)
        # pca = PCA(n_components='mle')
        # pca.fit(X)
        # X = pca.transform(X)

        cv_results = cross_validate(self.clf, X, y, cv=5)
        print("Cross Val Scores: {}".format(cv_results['test_score']))
        print("-" * 80)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        print("Training Model")
        print("-" * 80)
        self.model = self.clf.fit(X_train, y_train)

        return self.model

    def save(self):
        print("Saving Model")
        print("-" * 80)
        dump(self.model, os.path.join(self.output_path, self.model_file_name))

    def load(self, path: str):
        self.model = load(path)

        return self.model

    def predict(self, X_test, y_test):
        print("Predicting")
        print("-" * 80)
        self.pred = self.model.predict(X_test)
        self.gold = y_test

        return self.pred, self.gold

    def eval(self, X_test):
        print("Evaluating")
        print("-" * 80)
        df_stats = X_test
        df_stats["Model_Preds"] = np.where(self.pred == 1, "Over", "Under")
        df_stats["True_Gold"] = np.where(self.gold == 1, "Over", "Under")
        df_stats["Model_Correct"] = np.where(df_stats["Model_Preds"] == df_stats["True_Gold"], 1, 0)
        df_stats.to_csv(os.path.join(self.output_path, self.model_file_name + "_model_output.csv"))

    def calc_stats(self):
        df_stats = pd.DataFrame()
        tn, fp, fn, tp = confusion_matrix(self.pred, self.gold).ravel()
        print("Total Correct: {}".format(tp + tn))
        print("Total Incorrect: {}".format(fp + fn))
        print("Accuracy: {}".format((tp + tn) / (tp + tn + fp + fn)))

    def validate(self):
        df_val = pd.read_csv("/Users/administrator/PycharmProjects/NFL_O_U/output/nfl_results_2020.csv")
        df_val_one = df_val[df_val[""]]
        pass


if __name__ == '__main__':
    total_features = pd.read_csv("/Users/administrator/PycharmProjects/NFL_O_U/output/total_features.csv")
    m = Model(total_features)
    X_train, X_test, y_train, y_test = m.split_data()
    m.train(X_train, y_train)
    m.save()
    m.predict(X_test, y_test)
    m.eval(X_test)
    m.calc_stats()
