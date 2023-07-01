import pandas as pd
import sklearn.metrics as metrics
import plotly.express as px
import statsmodels.tools as smtools
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import plotly.tools as tls

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt

class SklearnLinearRegression:
    """
    Linear regression model implemented using scikit-learn.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe.
    features : list of str
        The list of feature column names.
    target : str
        The target column name.

    Attributes
    ----------
    target : str
        The target column name.
    data_frame : pandas.DataFrame
        The input dataframe.
    columns : list of str
        The list of feature column names.
    data_frame_feats : pandas.DataFrame
        The dataframe with only the feature columns.
    data_frame_target : pandas.Series
        The series with the target column values.
    x_train : pandas.DataFrame
        The training set features.
    x_test : pandas.DataFrame
        The test set features.
    y_train : pandas.Series
        The training set target.
    y_test : pandas.Series
        The test set target.
    lr : sklearn.linear_model.LinearRegression
        The linear regression model.
    y_pred_train : numpy.ndarray
        The predicted target values for the training set.
    y_pred_test : numpy.ndarray
        The predicted target values for the test set.
    """
    def __init__(self, dataframe, features: list[str], target: str):
        """
        Initialize the SklearnLinearRegression class.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input DataFrame containing the data.
        features : list of str
            The list of feature column names.
        target : str
            The target column name.

        """
        self.target = target
        self.data_frame = dataframe
        self.data_frame["Entry_Price"].fillna(0, inplace=True)
        self.data_frame["ema"].fillna(0, inplace=True)
        self.data_frame["Take_Profit"].fillna(99999, inplace=True)
        self.data_frame["Stop_Loss"].fillna(-99999, inplace=True)

        self.columns = features
        self.data_frame_feats = self.data_frame[self.columns]
        self.data_frame_target = self.data_frame[target]

        x = self.data_frame_feats
        y = self.data_frame_target

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.5
        )

        self.lr = LinearRegression()
        self.lr.fit(self.x_train, self.y_train)

        self.y_pred_train = self.lr.predict(self.x_train)
        self.y_pred_test = self.lr.predict(self.x_test)

    @property
    def results_evaul(self):
        """
        Evaluate the results of the linear regression model.

        Prints the evaluation metrics for both the training and test sets.
        """
        MAE_train = metrics.mean_absolute_error(self.y_train, self.y_pred_train)
        R2_train = metrics.r2_score(self.y_train, self.y_pred_train)
        RMSE_train = sqrt(metrics.mean_squared_error(self.y_train, self.y_pred_train))

        print("----- Train Evaluation -----")
        print(f"MAE: {round(MAE_train, 2)}")
        print(f"R2: {round(R2_train, 2)}")
        print(f"RMSE: {round(RMSE_train, 2)}")

        MAE_test = metrics.mean_absolute_error(self.y_test, self.y_pred_test)
        R2_test = metrics.r2_score(self.y_test, self.y_pred_test)
        RMSE_test = sqrt(metrics.mean_squared_error(self.y_test, self.y_pred_test))
        MAE_base_ratio = (
            metrics.mean_absolute_error(self.y_test, self.y_pred_test)
            / self.y_test.mean()
            * 100,
        )

        print("\n----- Test Evaluation -----")
        print(f"MAE: {round(MAE_test, 2)}")
        print(f"R2: {round(R2_test, 2)}")
        print(f"RMSE: {round(RMSE_test, 2)}")
        print(f"Result Mean: {round(self.y_test.mean(), 2)} \n")
        print(
            f"The percentage of MAE in relation to the mean of the base:"
            f"{round(MAE_base_ratio,2)}"
        )

        print(f"Test MAE: {round(MAE_test, 2)}")
