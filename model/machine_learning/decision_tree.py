from typing import Any
import pandas as pd
import numpy as np
import plotly.express as px

from pydotplus import graphviz
from sklearn import metrics, tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.core.display import Image


class DecisionTreeClassifier:
    def __init__(self, dataframe, tick_size):
        self.data_frame = dataframe
        self.data_frame.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            },
            inplace=True,
        )

        self.data_frame["Pips"] = (
            self.data_frame["Close"] - self.data_frame["Close"].shift(1)
        ) / tick_size

    def create_simple_variable(self, periods):
        self.data_frame["Return"] = self.data_frame["Close"].pct_change(periods)

        self.data_frame["Target"] = self.data_frame.Return.shift(-periods)

        self.data_frame["Target_cat"] = np.where(
            self.data_frame.Target > 0,
            "Bullish",
            "Bearish",
        )
        return self.data_frame

    def get_proportion(self, column):
        return self.data_frame[column].value_counts(normalize=True) * 100

    def create_variables(self):
        self.data_frame["std5"] = self.data_frame["Close"].rolling(5).std()
        self.data_frame["std10"] = self.data_frame["Close"].rolling(10).std()
        self.data_frame["prop"] = (
            self.data_frame["Close"] - self.data_frame["Open"]
        ) / (self.data_frame["High"] - self.data_frame["Low"])
        self.data_frame["dir_D"] = np.where(
            self.data_frame["Close"] > self.data_frame["Open"], "1", "0"
        )
        self.data_frame["dir_D-1"] = self.data_frame["dir_D"].shift(1)
        self.data_frame["dir_D-2"] = self.data_frame["dir_D"].shift(2)
        self.data_frame["dir_D-3"] = self.data_frame["dir_D"].shift(3)

        self.data_frame.dropna(
            axis=0, how="any", subset=self.data_frame.columns[-7:], inplace=True
        )

        return self

    def split_data(self, features_columns: list[str], target_column: list[str]):
        start_train = 0
        end_train = int(self.data_frame.shape[0] / 2)

        start_test = end_train
        end_test = self.data_frame.shape[0]

        df1_train1 = self.data_frame[start_train:end_train]
        self.data_frame = self.data_frame[start_test:end_test]

        x_train1 = df1_train1[features_columns]
        y_train1 = df1_train1[target_column]

        x_test1 = self.data_frame[features_columns]
        y_test1 = self.data_frame[target_column]
        return x_train1, x_test1, y_train1, y_test1

    def decision_tree_classifier(self, features_columns, target_column):
        x_train1, x_test1, y_train1, y_test1 = self.split_data(
            features_columns, target_column
        )

        decision_tree_classifier = tree.DecisionTreeClassifier()

        decision_tree_classifier.fit(x_train1, y_train1)
        y_pred_test1 = decision_tree_classifier.predict(x_test1)

        return y_test1, y_pred_test1, decision_tree_classifier

    def results_report(self, y_test1, y_pred_test1):
        accuracy = round(metrics.accuracy_score(y_test1, y_pred_test1), 3) * 100

        print(f"{confusion_matrix(y_test1, y_pred_test1)} \n")
        print("--------------------------------------------------------------")
        print(f"\n {classification_report(y_test1, y_pred_test1)}")
        print(f"Accuracy:{accuracy}")

    def create_tree_png(
        self,
        clf: tree.DecisionTreeClassifier,
        feature_names: list["str"],
        class_names: list["str"],
    ):
        dot_data = StringIO()

        export_graphviz(
            clf,
            out_file=dot_data,
            filled=True,
            rounded=True,
            special_characters=True,
            feature_names=feature_names,
            class_names=class_names,
        )

        graph = graphviz.graph_from_dot_data(dot_data.getvalue())
        graph: Any

        return Image(graph.create_png())
