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

    def __init__(self, features, target, test_size, shuffle = False, **kwargs):
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
                shuffle=shuffle,
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
        fig = px.line(
            x=fpr,
            y=tpr,
            title=f"ROC Curve (AUC={auc(fpr, tpr):.4f})",
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
            width=700,
            height=500,
        )

        fig.add_shape(
            type="line",
            line=dict(dash="dash"),
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            opacity=0.65,
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain="domain")
        return fig

    def get_results(
        self,
        returns: np.ndarray | pd.Series,
        predict_proba: None | np.ndarray,
        test_buy_conds: np.ndarray[bool],
        test_sell_conds: np.ndarray[bool],
        trading_fee: float = 0.03,
        log_result: bool = True,
        **kwargs,
    ) -> px.line:
        """
        Generate a Plotly Express line chart showing the cumulative
        total returns based on the strategy.

        Parameters:
        -----------
        returns : np.ndarray or pd.Series
            The array or Series of historical returns.
        predict_proba : None or np.ndarray, optional
            The predicted probabilities of the positive class, if
            available. If None, predictions will be made using the
            scikit-learn model.
        test_buy_conds : np.ndarray of bool
            Boolean conditions indicating buy signals.
        test_sell_conds : np.ndarray of bool
            Boolean conditions indicating sell signals.
        **kwargs : optional
            Additional keyword arguments to customize the appearance
            of the plot.

        Returns:
        --------
        plotly.graph_objs._figure.Figure
            The Plotly Express line chart displaying the cumulative
            total returns over time.
        """
        if predict_proba:
            y_pred_probs = np.copy(predict_proba)
        else:
            y_pred_probs = self.sk_model.predict_proba(self.X_test)[:, 1]

        trading_cost = (trading_fee * 2) / 100

        return_df = pd.DataFrame({"y_pred_probs" : y_pred_probs})
        return_df["Position"] = np.where(
            test_buy_conds,
            1,
            np.where(test_sell_conds, -1, 0)
        )

        return_df["Return"] = returns
        return_df["Return"] = return_df["Return"] / 100

        if log_result:
            return_df["Result"] = (
                return_df["Return"]
                * return_df["Position"]
                + 1
            )

            return_df["Liquid_Result"] = np.where(
                return_df["Position"] != 0,
                return_df["Return"] * return_df["Position"] - trading_cost + 1,
                1
            )

            return_df["Total_Return"] = return_df["Result"].cumprod()
            return_df["Liquid_Return"] = return_df["Liquid_Result"].cumprod()

        else:
            return_df["Result"] = np.where(
                return_df["Position"] != 0,
                np.log(return_df["Return"] * return_df["Position"] + 1) + 1,
                1,
            )

            return_df["Liquid_Result"] = np.log(
                return_df["Return"] * return_df["Position"] - trading_cost + 1
            ) + 1

            return_df["Total_Return"] = return_df["Result"].cumprod()
            return_df["Liquid_Return"] = return_df["Liquid_Result"].cumprod()

        return px.line(
            return_df,
            y=["Total_Return", "Liquid_Return"]
        ).update_layout(**kwargs)
