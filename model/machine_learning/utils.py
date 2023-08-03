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
