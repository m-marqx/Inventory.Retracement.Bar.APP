import pandas as pd
from typing import Literal
from model.utils.exceptions import InvalidArgumentError


class MathFeature:
    """
    A class for calculating mathematical features based on the input
    data.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing the data.
    source_column : str
        The name of the column representing the price values.
    feat_last_column : str, optional
        The name of the column representing the last feature
        (default: None).
    return_type : Literal["short", "full"], optional
        The return type of methods ("short" returns only calculated
        values, "full" returns the modified DataFrame with added
        columns).
        (default: "short")

    Attributes
    ----------
    dataframe : pandas.DataFrame
        The copy of the input dataframe.
    source_column : str
        The name of the column representing the price values.
    feat_last_column : str
        The name of the column representing the last feature.
    return_type : Literal["short", "full"]
        The return type of methods.

    Methods
    -------
    rolling_ratio(fast_length, slow_length, method)
        Calculate a rolling ratio of two rolling averages.

    ratio(length, method)
        Compute ratio-based features.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        source_column: str,
        feat_last_column: str = None,
        return_type: Literal["short", "full"] = "short",
    ) -> None:
        """
        Initialize the MathFeature class.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input dataframe containing the data.
        source_column : str
            The name of the column representing the price values.
        feat_last_column : str, optional
            The name of the column representing the last feature
            (default: None).
        return_type : Literal["short", "full"], optional
            Determines the return type of the class methods. If "short", only
            the calculated values are returned. If "full", the modified
            DataFrame with added columns is returned.
            (default: "short").
        """
        self.dataframe = dataframe.copy()
        self.source_column = source_column
        self.feat_last_column = feat_last_column
        self.return_type = return_type

        if return_type not in ["short", "full"]:
            raise InvalidArgumentError(f"{return_type} not found")

    def rolling_ratio(
        self,
        fast_length: int,
        slow_length: int,
        method: str,
    ) -> pd.DataFrame:
        """
        Calculate a rolling ratio feature based on the values of two
        rolling averages.

        Parameters:
        -----------
        fast_length : int
            The window size for the fast rolling average.
        slow_length : int
            The window size for the slow rolling average.
        method : str
            The pandas method used for rolling averages
            (e.g., "mean", "std", "median").

        Returns:
        --------
        pd.DataFrame
            The original DataFrame with an additional column for the
            rolling ratio.

        Raises:
        -------
        ValueError
            If an invalid method is specified.
        """
        if fast_length == slow_length:
            raise ValueError("fast_length and slow_length must be different")

        fast_rolling = self.dataframe[self.source_column].rolling(fast_length)
        slow_rolling = self.dataframe[self.source_column].rolling(slow_length)

        try:
            fast_rolling = getattr(fast_rolling, method)()
            slow_rolling = getattr(slow_rolling, method)()
        except AttributeError as exc:
            raise ValueError(f"Invalid method '{method}'") from exc

        rolling_std_ratio = fast_rolling / slow_rolling

        if self.return_type == "short":
            return rolling_std_ratio.rename(f'rolling_ratio_{method}')

        self.dataframe["rolling_std_ratio"] = rolling_std_ratio

        if self.feat_last_column:
            self.dataframe = self.dataframe.reorder_columns(
                self.feat_last_column, self.dataframe.columns[-1:]
            )

        return self.dataframe

    def ratio(self, length: int, method: str) -> pd.DataFrame:
        """
        Compute ratio-based features.

        Parameters:
        -----------
        length : int
            The window length for rolling statistics.
        method : str
            The method used for rolling averages
            (e.g., "mean", "std", "median").
        Returns:
        --------
        pd.DataFrame
            Returns the DataFrame with ratio-based features added.
        """
        rolling_data = self.dataframe[self.source_column].rolling(length)
        rolling_data = getattr(rolling_data, method)()

        ratio_variable = (
            self.dataframe[self.source_column]
            / rolling_data - 1
        )

        if self.return_type == "short":
            return ratio_variable.iloc[:, -1:]

        self.dataframe["ratio_variable"] = ratio_variable

        if self.feat_last_column:
            self.dataframe = self.dataframe.reorder_columns(
                self.feat_last_column, self.dataframe.columns[-1:]
            )

        return self.dataframe
