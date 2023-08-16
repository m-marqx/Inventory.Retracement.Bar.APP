from typing import Literal
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import plotly.express as px

class LogisticModel:
    """
    Perform logistic regression analysis using both statsmodels and
    scikit-learn.

    Parameters:
    -----------
    features : array-like
        The input features for the logistic regression.
    target : array-like
        The target values for the logistic regression.
    test_size : float
        The proportion of the dataset to include in the test split.
    **kwargs : optional
        Additional keyword arguments to be passed to the
        `train_test_split` function.

    Attributes:
    -----------
    features : array-like
        The input features for the logistic regression.
    target : array-like
        The target values for the logistic regression.
    model : None
        Placeholder for the trained model.
    X_train : array-like
        The training input features.
    X_test : array-like
        The testing input features.
    y_train : array-like
        The training target values.
    y_test : array-like
        The testing target values.
    sm_model : statsmodels.GLM
        The statsmodels logistic regression model.
    sk_model : sklearn.linear_model.LogisticRegression
        The scikit-learn logistic regression model.

    Methods:
    --------
    model_predict(
    method: Literal["statsmodels", "sklearn"], threshold: float
    ) -> tuple
        Predict the target values using the specified method and
        calculate accuracy.
    sk_auc_stats() -> tuple
        Calculate the ROC curve, AUC, and Gini coefficient for the
        scikit-learn model.
    fpr_tpr_curve(
    fpr: np.ndarray | pd.Series, tpr: np.ndarray | pd.Series
    ) -> px.line
        Generate a Plotly Express line chart for FPR vs. TPR curve.
    """

    def __init__(self, features, target, test_size, **kwargs):
        """
        Initialize the CalculateLogisticRegression object.

        Parameters:
        -----------
        features : array-like
            The input features for training the logistic regression
            model.
        target : array-like
            The target values for training the logistic regression
            model.
        test_size : float
            The proportion of the dataset to include in the test
            split.
        **kwargs : additional keyword arguments
            Additional keyword arguments to be passed to the
            `train_test_split` function.
        """
        self.features = features
        self.target = target
        self.model = None

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                self.features,
                self.target,
                test_size=test_size,
                **kwargs
            )
        )

        self.sm_model = sm.GLM(
            self.y_train,
            self.X_train,
            family=sm.families.Binomial()
        )

        self.sk_model = LogisticRegression().fit(self.X_train, self.y_train)

