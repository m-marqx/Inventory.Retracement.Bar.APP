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
    """
    Decision Tree Classifier for predicting the direction of financial
    market movements.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing financial market data.
    tick_size : int
        The tick size used for calculating Pips.

    Attributes
    ----------
    data_frame : pandas.DataFrame
        The input dataframe containing financial market data with
        columns renamed to ['Open', 'High', 'Low', 'Close'].
    tick_size : int
        The tick size used for calculating Pips.

    Methods
    -------
    create_simple_variable(periods):
        Create simple variables 'Return' and 'Target' for training
        the classifier.

    get_proportion(column):
        Get the proportion of values in a specific column.

    create_variables():
        Create additional variables for training the classifier.

    split_data(features_columns, target_column):
        Split the data into training and testing sets.

    decision_tree_classifier(features_columns, target_column):
        Train a decision tree classifier on the training data and make
        predictions on the test data.

    results_report(y_test1, y_pred_test1):
        Generate a report of classification results.

    create_tree_png(clf, feature_names, class_names):
        Create a PNG image of the decision tree.

    get_results(y_pred_test):
        Calculate and return the cumulative return based on classifier
        predictions.

    plot_returns(data, title, x_title, y_title):
        Plot the cumulative returns.

    custom_decision_tree_classifier(decision_tree_classifier,
    features_columns, target_column):
        Train a custom decision tree classifier on the training data.

    generate_aleatory_results(column_name):
        Generate aleatory results for a specific column.

    """
    def __init__(self, dataframe, tick_size):
        """
        Initialize the DecisionTreeClassifier.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input dataframe containing financial market data.
        tick_size : int
            The tick size used for calculating Pips.
        """
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
        """
        Create simple variables 'Return' and 'Target' for training the
        classifier.

        Parameters
        ----------
        periods : int
            The number of periods for calculating returns and target.

        Returns
        -------
        pandas.DataFrame
            The modified dataframe with additional columns 'Return' and
            'Target'.
        """
        self.data_frame["Return"] = self.data_frame["Close"].pct_change(periods)

        self.data_frame["Target"] = self.data_frame.Return.shift(-periods)

        self.data_frame["Target_cat"] = np.where(
            self.data_frame.Target > 0,
            "Bullish",
            "Bearish",
        )
        return self.data_frame

    def get_proportion(self, column):
        """
        Get the proportion of values in a specific column.

        Parameters
        ----------
        column : str
            The column for which the proportion is calculated.

        Returns
        -------
        pandas.Series
            The proportion of each unique value in the specified column.
        """
        return self.data_frame[column].value_counts(normalize=True) * 100

    def create_variables(self):
        """
        Create additional variables for training the classifier.

        Returns
        -------
        DecisionTreeClassifier
            The updated instance of the DecisionTreeClassifier.
        """
        self.data_frame["std5"] = self.data_frame["Close"].rolling(5).std()
        self.data_frame["std10"] = self.data_frame["Close"].rolling(10).std()
        self.data_frame["candle_ratio"] = (
            self.data_frame["Close"] - self.data_frame["Open"]
        ) / (self.data_frame["High"] - self.data_frame["Low"])
        self.data_frame["dir_S"] = np.where(
            self.data_frame["Close"] > self.data_frame["Open"], "1", "0"
        )
        self.data_frame["dir_S-1"] = self.data_frame["dir_S"].shift(1)
        self.data_frame["dir_S-2"] = self.data_frame["dir_S"].shift(2)
        self.data_frame["dir_S-3"] = self.data_frame["dir_S"].shift(3)

        self.data_frame.dropna(
            axis=0, how="any", subset=self.data_frame.columns[-7:], inplace=True
        )

        return self

    def split_data(self, features_columns: list[str], target_column: list[str]):
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        features_columns : list of str
            The list of feature column names.
        target_column : str
            The name of the target column.

        Returns
        -------
        tuple
            A tuple containing the training and testing data
            (x_train1, x_test1, y_train1, y_test1).
        """
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
        """
        Train a decision tree classifier on the training data and make
        predictions on the test data.

        Parameters
        ----------
        features_columns : list of str
            The list of feature column names.
        target_column : str
            The name of the target column.

        Returns
        -------
        tuple
            A tuple containing the true labels (y_test1), predicted
            labels (y_pred_test1), and the trained classifier.
        """
        x_train1, x_test1, y_train1, y_test1 = self.split_data(
            features_columns, target_column
        )

        decision_tree_classifier = tree.DecisionTreeClassifier()

        decision_tree_classifier.fit(x_train1, y_train1)
        y_pred_test1 = decision_tree_classifier.predict(x_test1)

        return y_test1, y_pred_test1, decision_tree_classifier

    def results_report(self, y_test1, y_pred_test1):
        """
        Generate a report of classification results.

        Parameters
        ----------
        y_test1 : array-like
            The true labels.
        y_pred_test1 : array-like
            The predicted labels.
        """
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
        """
        Create a PNG image of the decision tree.

        Parameters
        ----------
        clf : sklearn.tree.DecisionTreeClassifier
            The trained decision tree classifier.
        feature_names : list of str
            The list of feature names.
        class_names : list of str
            The list of class names.

        Returns
        -------
        PIL.Image.Image
            The PNG image of the decision tree.
        """
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

    def get_results(self, y_pred_test):
        """
        Calculate and return the cumulative return based on classifier
        predictions.

        Parameters
        ----------
        y_pred_test : array-like
            The predicted labels.

        Returns
        -------
        pandas.DataFrame
            The dataframe with additional columns 'Ret_Pips' and
            'Ret_Pips_Total'.
        """
        dtc_data_frame = pd.DataFrame({"Predicted": y_pred_test})

        dtc_data_frame["Ret_Pips"] = np.where(
            dtc_data_frame["Predicted"] == "Bullish",
            self.data_frame["Pips"],
            "0",
        )

        dtc_data_frame["Ret_Pips"] = np.where(
            dtc_data_frame["Predicted"] == "Bearish",
            -1 * self.data_frame["Pips"],
            dtc_data_frame["Ret_Pips"],
        )

        dtc_data_frame["Ret_Pips"] = dtc_data_frame["Ret_Pips"].astype(float)

        dtc_data_frame["Ret_Pips_Total"] = (
            dtc_data_frame["Ret_Pips"]
            .cumsum()
        )

        return dtc_data_frame

    def plot_returns(self, data, title, x_title, y_title):
        """
        Plot the cumulative returns.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataframe containing the cumulative returns.
        title : str
            The title of the plot.
        x_title : str
            The title of the x-axis.
        y_title : str
            The title of the y-axis.

        Returns
        -------
        plotly.graph_objects.Figure
            The plotly figure object.
        """
        return px.line(data).update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
        )

    def custom_decision_tree_classifier(
        self,
        decision_tree_classifier: tree.DecisionTreeClassifier,
        features_columns,
        target_column,
    ):
        """
        Train a custom decision tree classifier on the training data.

        Parameters
        ----------
        decision_tree_classifier : sklearn.tree.DecisionTreeClassifier
            The custom decision tree classifier.
        features_columns : list of str
            The list of feature column names.
        target_column : str
            The name of the target column.

        Returns
        -------
        tuple
            A tuple containing the predicted labels (y_pred_test1) and
            the trained classifier.
        """
        x_train1, x_test1, y_train1, _ = self.split_data(
            features_columns, target_column
        )

        decision_tree_classifier.fit(x_train1, y_train1)
        y_pred_test1 = decision_tree_classifier.predict(x_test1)

        return y_pred_test1, decision_tree_classifier

    def generate_aleatory_results(self, column_name: str):
        """
        Generate aleatory results for a specific column.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        DecisionTreeClassifier
            The updated instance of the DecisionTreeClassifier.
        """
        self.data_frame[column_name] = np.random.randint(
            0, 2, size=len(self.data_frame.shape[0])
        )

        self.data_frame[column_name] = np.where(
            self.data_frame[column_name] == 1, self.data_frame["Pips"], "0"
        )

        self.data_frame[column_name] = np.where(
            self.data_frame[column_name] == 0,
            -1 * self.data_frame["Pips"],
            self.data_frame[column_name],
        )

        self.data_frame[column_name] = (
            self.data_frame[column_name]
            .cumsum()
        )

        return self
