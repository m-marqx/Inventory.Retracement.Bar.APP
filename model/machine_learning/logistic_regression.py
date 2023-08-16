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

    def model_predict(
        self,
        method: Literal["statsmodels", "sklearn"],
        threshold
    ) -> tuple:
        """
        Predict using the trained logistic regression model.

        Parameters:
        -----------
        method : {"statsmodels", "sklearn"}
            The method to use for prediction.
        threshold : float
            The threshold to apply for classification.

        Returns:
        --------
        tuple
            A tuple containing predicted classes and accuracy.
        """
        if method == "statsmodels":
            y_pred = self.sm_model.predict(self.X_test)
            y_pred_classes = np.where(y_pred > threshold, 1, 0)
            accuracy = np.mean(y_pred_classes == self.y_test)
            return y_pred_classes, accuracy

        if method == "sklearn":
            y_pred = self.sk_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            return y_pred, accuracy
        raise ValueError("method must be either 'statsmodels' or 'sklearn'")

    @property
    def sk_auc_stats(self) -> tuple:
        """
        Calculate AUC and Gini coefficient using sklearn.

        Returns:
        --------
        tuple
            A tuple containing False Positive Rate (FPR), True Positive
            Rate (TPR), ROC AUC, and Gini coefficient.
        """
        y_pred_probs = self.sk_model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        gini = 2 * roc_auc - 1
        return fpr, tpr, roc_auc, gini

    def fpr_tpr_curve(
        self,
        fpr: np.ndarray | pd.Series,
        tpr: np.ndarray | pd.Series
    ) -> px.line:
        """
        Generate a False Positive Rate (FPR) vs. True Positive Rate
        (TPR) curve.

        Parameters:
        -----------
        fpr : np.ndarray or pd.Series
            The False Positive Rate values.
        tpr : np.ndarray or pd.Series
            The True Positive Rate values.

        Returns:
        --------
        px.line
            A Plotly line plot representing the FPR vs. TPR curve.
        """
        return px.line(y=[tpr , fpr], x=fpr)
