import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz

class DataPreprocessor:
    """
    A class for preprocessing and splitting data for binary
    classification tasks.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data.
    features : str or list of str, optional
        The list of feature column names to use in the DataFrame.
        If None, defaults to ["IRB_Condition", "Signal", "uptrend"].

    Attributes:
    -----------
    dataframe : pd.DataFrame
        The DataFrame containing the data.
    features : list of str
        The list of feature column names used in the DataFrame.

    Methods:
    --------
    get_target_bin(periods=12, column="close", cutoff=None) ->
    pd.DataFrame
        Generate a target binary DataFrame based on a specified column.

    get_splits(target) ->
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Split the DataFrame into training and testing sets.

    tree_view(target, fitted_tree) -> None
        Visualize the decision tree using Graphviz.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: str | list[str] = None,
    ) -> None:
        """
        Initialize the DataPreprocessor object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input DataFrame containing the data.
        features : str or list of str, optional
            The list of feature column names to use in the DataFrame.
            If None, defaults to ["IRB_Condition", "Signal", "uptrend"].
        """
        self.dataframe = dataframe.copy()

        if features is None:
            features = ["IRB_Condition", "Signal", "uptrend"]

        self.features = list(features)

    def get_target_bin(
        self,
        periods: int = 12,
        column: str = "close",
        cutoff: float | list[float] = None,
    ) -> pd.DataFrame:
        """
        Generate a target binary DataFrame based on a specified column.

        Parameters:
        -----------
        periods : int, optional
            The number of periods used to calculate the percentage
            change.
            (default: 12)
        column : str, optional
            The column name from which the target binary is generated.
            (default: "close")
        cutoff : float or list of float, optional
            The cutoff value(s) to determine the target binary values.
            If a list of float values is provided, multiple target bins
            will be generated.
            If None, defaults to [0.25, 0.30, 0.35]

        Returns:
        --------
        pd.DataFrame
            The DataFrame with binary targets.
        """
        if cutoff is None:
            cutoff = [0.25, 0.30, 0.35]

        features = [column] + list(self.features)
        data_frame = self.dataframe[features].astype("float64").copy()

        data_frame["return"] = data_frame[column].pct_change(periods)
        data_frame["target"] = data_frame["return"].shift(-periods)

        for value in enumerate(cutoff):
            cutoff_value = data_frame["target"].std() * value[1]
            data_frame[f"Target_Bin{value[0]}"] = np.where(
                data_frame["target"] > cutoff_value,
                1,
                np.where(data_frame["target"] < -cutoff_value, -1, 0),
            )

        return data_frame

    def get_splits(
        self,
        target: list | str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into training and testing sets.

        Parameters:
        -----------
        target : list or str
            The target column name(s) to use for generating the training
            and testing sets.

        Returns
        -------
        tuple of pd.DataFrame
            The tuple containing training data, training target,
            testing data, and testing target.
        """
        end_train_index = int(self.dataframe.shape[0] / 2)

        x_train = self.dataframe.iloc[:end_train_index]
        y_train = pd.DataFrame()
        x_test = self.dataframe.iloc[end_train_index:]
        y_test = pd.DataFrame()

        df_train = x_train.loc[:, self.features]
        df_test = x_test.loc[:, self.features]

        for value in enumerate(target):
            y_train[f"target_{value[0]}"] = x_train[value[1]]
            print(y_train.shape)

        for value in enumerate(target):
            y_test[f"target_{value[0]}"] = x_test[value[1]]
            print(y_test.shape)

        return df_train, y_train, df_test, y_test

    def tree_view(
        self,
        target: pd.Series,
        fitted_tree: RandomForestClassifier,
    ) -> None:
        """
        Visualize the decision tree using Graphviz.

        Parameters:
        -----------
        target : pd.Series
            The target column to be used in the decision tree
            visualization.
        fitted_tree : RandomForestClassifier
            The fitted RandomForestClassifier model containing
            the decision tree.

        Returns
        -------
        None
            The tree visualization is displayed and saved as a PNG
            image.
        """
        tree = fitted_tree.estimators_[0]
        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=self.features,
            class_names=target.astype("str"),
            filled=True,
            rounded=True,
            special_characters=True,
        )

        graphviz.Source(dot_data, format="png").view()
