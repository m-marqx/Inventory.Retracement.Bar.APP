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
        """Calculate the fast and slow exponential moving averages."""
        self.fast_ma = ma.ema(self.source, self.fast_length)
        self.slow_ma = ma.ema(self.source, self.slow_length)

    def __set_sma(self):
        """Calculate the fast and slow simple moving averages."""
        self.fast_ma = ma.sma(self.source, self.fast_length)
        self.slow_ma = ma.sma(self.source, self.slow_length)

    @property
    def get_histogram(self) -> pd.DataFrame:
        """Calculate the MACD histogram.

        Returns
        -------
        pandas.DataFrame
            The MACD histogram.

        """
        macd = (self.fast_ma - self.slow_ma).dropna()
        macd_signal = ma.ema(macd, self.signal_length)
        histogram = macd - macd_signal
        return histogram
