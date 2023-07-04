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

