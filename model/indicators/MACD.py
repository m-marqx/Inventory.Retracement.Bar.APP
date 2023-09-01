from typing import Literal
import pandas as pd
from model.indicators import MovingAverage

ma = MovingAverage()


class MACD:
    """Moving Average Convergence Divergence (MACD) indicator.

    Calculates the MACD histogram based on the provided source data and parameters.

    Parameters
    ----------
    source : pandas.Series
        The source data.
    fast_length : int
        The length of the fast moving average.
    slow_length : int
        The length of the slow moving average.
    signal_length : int
        The length of the signal line moving average.
    method : str, optional
        The method used for calculating moving averages. Defaults to "ema".
        Supported methods are "sma" (Simple Moving Average) and "ema"
        (Exponential Moving Average).

    Attributes
    ----------
    fast_ma : pandas.Series
        The fast moving average.
    slow_ma : pandas.Series
        The slow moving average.

    """
    def __init__(
        self,
        source: pd.Series,
        fast_length: int,
        slow_length: int,
        signal_length: int,
        method: Literal["ema", "sma"] = "ema",
    ):
        """
        Initialize the Moving Average Convergence Divergence (MACD)
        indicator.

        Parameters:
        -----------
        source : pd.Series
            The input time series data for calculating MACD.
        fast_length : int
            The number of periods for the fast moving average.
        slow_length : int
            The number of periods for the slow moving average.
        signal_length : int
            The number of periods for the signal line moving average.
        method : Literal["ema", "sma"], optional
            The method to use for calculating moving averages, either
            "ema" for Exponential Moving Average or "sma" for Simple
            Moving Average.
            (default: "ema")

        Raises:
        -------
        ValueError
            If an invalid method is provided.
        """
        self.source = source
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length
        if method == "sma":
            self.__set_sma()
        if method == "ema":
            self.__set_ema()
        else:
            raise ValueError(f"'{method}' is not a valid method.")

    def __set_ema(self):
        self.fast_ma = ma.ema(self.source, self.fast_length)
        self.slow_ma = ma.ema(self.source, self.slow_length)

    def __set_sma(self):
        self.fast_ma = ma.sma(self.source, self.fast_length)
        self.slow_ma = ma.sma(self.source, self.slow_length)

    @property
    def get_histogram(self) -> pd.DataFrame:
        """
        Calculate the MACD histogram.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the MACD histogram values.
        """
        macd = (self.fast_ma - self.slow_ma).dropna()
        macd_signal = ma.ema(macd, self.signal_length)
        histogram = macd - macd_signal
        return histogram
