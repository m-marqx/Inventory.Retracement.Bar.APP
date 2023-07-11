import pandas as pd
from model.indicators import MovingAverage

ma = MovingAverage()


class MACD:
    def __init__(self, source: pd.Series, fast_length: int, slow_length: int, signal_length: int):
        """
        Initialize the MACD object.

        Parameters:
        -----------
        source : pd.Series
            The input time series data.
        fast_length : int
            The number of periods for the fast EMA.
        slow_length : int
            The number of periods for the slow EMA.
        signal_length : int
            The number of periods for the signal EMA.
        """
        self.source = source
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length

    def set_ema(self):
        """
        Set the Exponential Moving Average (EMA) for the MACD calculation.

        Returns:
        --------
        MACD
            The MACD object.
        """
        self.fast_ma = ma.ema(self.source, self.fast_length)
        self.slow_ma = ma.ema(self.source, self.slow_length)

        return self

    def set_sma(self):
        """
        Set the Simple Moving Average (SMA) for the MACD calculation.

        Returns:
        --------
        MACD
            The MACD object.
        """
        self.fast_ma = ma.sma(self.source, self.fast_length)
        self.slow_ma = ma.sma(self.source, self.slow_length)

        return self

    def MACD(self) -> pd.DataFrame:
        """
        Calculate MACD using the specified method.

        Returns:
        --------
        pd.DataFrame
            The calculated MACD data as a DataFrame.
        """
        self.MACD = self.fast_ma - self.slow_ma
        self.data_frame = pd.DataFrame({"MACD": self.MACD}).dropna(axis=0)
        self.data_frame["MACD_Signal"] = ma.ema(self.data_frame["MACD"], self.signal_length)
        self.data_frame["Histogram"] = self.data_frame["MACD"] - self.data_frame["MACD_Signal"]

        return self.data_frame
