import pandas as pd
import numpy as np
from model.indicators import MovingAverage

ma = MovingAverage()


class DMI:
    """
    Attributes:
    -----------
    source : pd.Series
        The source values from the DataFrame.
    high : pd.Series
        The high prices from the DataFrame.
    low : pd.Series
        The low prices from the DataFrame.
    length : int
        The lookback period for calculating the Stochastic Oscillator.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        source: str,
        high: str = None,
        low: str = None,
        length: int = 14
    ) -> None:
        """
        Initialize the DMI object with the given data and
        parameters.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame containing the source, high, and low data.
        source : str
            The column name in the DataFrame representing the source
            data.
        high : str, optional
            The column name in the DataFrame representing the high
            data. If not provided,
            it will be inferred from common column names.
        low : str, optional
            The column name in the DataFrame representing the low
            data. If not provided,
            it will be inferred from common column names.
        length : int, optional
            The length of the stochastic period. Default is 14.
        """
        self.source = dataframe[source]

        if high is None:
            if "High" in dataframe.columns:
                self.high = dataframe["High"]
            elif "high" in dataframe.columns:
                self.high = dataframe["high"]
        else:
            self.high = dataframe[high]

        if low is None:
            if "Low" in dataframe.columns:
                self.low = dataframe["Low"]
            elif "low" in dataframe.columns:
                self.low = dataframe["low"]
        else:
            self.low = dataframe[low]

        self.length = length
