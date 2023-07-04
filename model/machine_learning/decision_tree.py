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
