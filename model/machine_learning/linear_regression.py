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

    @property
    def coeficients(self):
        """
        Get the coefficients of the linear regression model.

        Returns
        -------
        pandas.DataFrame
            A dataframe with the coefficients and corresponding feature names.
        """
        self.lr.coef_

        coef = pd.DataFrame(self.lr.coef_, self.columns)
        coef.columns = ["Coeficientes"]

        return coef

    @property
    def get_fig_results(self):
        """
        Generate a scatter plot of the predicted vs. actual target values.

        Returns
        -------
        plotly.graph_objects.Figure
            The scatter plot figure.
        """
        fig = px.scatter(
            x=self.y_test,
            y=self.y_pred_test,
            labels={"x": f"{self.target} real", "y": f"{self.target} expected"},
        )
        fig.update_traces(
            marker=dict(color="blue", symbol="x"),
            mode="markers",
            name="Real x Expected",
        )
        fig.update_layout(
            title=f"{self.target} - Linear Regression",
            xaxis_title=f"{self.target} previsto",
            yaxis_title=f"{self.target} Real",
        )
        return fig


class StatsmodelsLinearRegression:
    """
    Linear regression model implemented using statsmodels.

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
    x_train_np : numpy.ndarray
        The training set features as a numpy array.
    y_train_np : numpy.ndarray
        The training set target as a numpy array.
    x_test_np : numpy.ndarray
        The test set features as a numpy array.
    x_train_const : numpy.ndarray
        The training set features with added constant term.
    x_test_const : numpy.ndarray
        The test set features with added constant term.
    lr_sm : statsmodels.regression.linear_model.OLS
        The ordinary least squares linear regression model.
    y_pred_train_sm : numpy.ndarray
        The predicted target values for the training set.
    y_pred_test_sm : numpy.ndarray
        The predicted target values for the test set.
    """
    def __init__(self, dataframe, features: list[str], target: str):
        """
        Initialize the StatsmodelsLinearRegression class.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input dataframe.
        features : list of str
            The list of feature column names.
        target : str
            The target column name.

        """
        self.data_frame = dataframe.copy()

        if "Take_Profit" in features:
            self.data_frame["Take_Profit"].fillna(99999, inplace=True)

        if "Stop_Loss" in features:
            self.data_frame["Stop_Loss"].fillna(-99999, inplace=True)

        self.data_frame.fillna(0, inplace=True)

        self.columns = features
        self.data_frame_feats = self.data_frame[self.columns]
        self.data_frame_target = self.data_frame[target]

        x = self.data_frame_feats
        y = self.data_frame_target

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.5
        )

        self.x_train_np = np.asarray(self.x_train)
        self.y_train_np = np.asarray(self.y_train)
        self.x_test_np = np.asarray(self.x_test)

        self.x_train_np = np.nan_to_num(self.x_train_np)
        self.y_train_np = np.nan_to_num(self.y_train_np)
        self.x_test_np = np.nan_to_num(self.x_test_np)

        self.x_train_const = sm.add_constant(self.x_train_np)
        self.x_test_const = sm.add_constant(self.x_test_np)

        self.x_train_const = self.x_train_const.astype(np.float64)
        self.y_train_np = self.y_train_np.astype(np.float64)

        self.lr_sm = sm.OLS(self.y_train_np, self.x_train_const).fit()

        self.y_pred_train_sm = self.lr_sm.predict(self.x_train_const)
        self.y_pred_test_sm = self.lr_sm.predict(self.x_test_const)

    @property
    def summary(self):
        """
        Get the summary of the linear regression model.

        Returns
        -------
        statsmodels.iolib.summary.Summary
            The summary object containing model statistics.
        """
        return self.lr_sm.summary()

    @property
    def results_eval(self):
        """
        Evaluate the results of the linear regression model.

        Prints the evaluation metrics for both the training and test sets.
        """
        MAE_train_sm = smtools.eval_measures.meanabs(self.y_train, self.y_pred_train_sm)
        R2_train_sm = self.lr_sm.rsquared
        RMSE_train_sm = smtools.eval_measures.rmse(self.y_train, self.y_pred_train_sm)

        print("----- Train Evaluation -----")
        print("MAE: ", round(MAE_train_sm, 2))
        print("R2: ", round(R2_train_sm, 2))
        print("RMSE: ", round(RMSE_train_sm, 2))

        MAE_test_sm = smtools.eval_measures.meanabs(self.y_test, self.y_pred_test_sm)
        RMSE_test_sm = smtools.eval_measures.rmse(self.y_test, self.y_pred_test_sm)

        print("\n----- Test Evaluation -----")
        print("MAE: ", round(MAE_test_sm, 2))
        print("RMSE: ", round(RMSE_test_sm, 2))

