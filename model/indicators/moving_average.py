import pandas as pd

# Indicadores
class MovingAverage:
    """
    A class for calculating Simple Moving Average (SMA)
    and Exponential Moving Average (EMA) of time series data.

    Attributes:
    -----------
    None

    Methods:
    --------
    sma(source: pd.Series, length: int) -> pd.Series:
        Calculate the Simple Moving Average (SMA)
        of the input time series data.

    ema(source: pd.Series, length: int) -> pd.Series:
        Calculate the Exponential Moving Average (EMA)
        of the input time series data.
    """

    def sma(self, source: pd.Series, length: int) -> pd.Series:
        """
        Calculate the Simple Moving Average (SMA)
        of the input time series data.

        Parameters:
        -----------
        source : pd.Series
            The time series data to calculate the SMA for.
        length : int
            The number of periods to include in the SMA calculation.

        Returns:
        --------
        pd.Series
            The calculated SMA time series data.
        """
        sma = source.rolling(length).mean()
        return sma.dropna(axis=0)

    def ema(self, source: pd.Series, length: int) -> pd.Series:
        """
        Calculate the Exponential Moving Average (EMA)
        of the input time series data.

        Parameters:
        -----------
        source : pandas.Series
            The time series data to calculate the EMA for.
        length : int
            The number of periods to include in the EMA calculation.

        Returns:
        --------
        pandas.Series
            The calculated EMA time series data.
        """
        sma = source.rolling(window=length, min_periods=length).mean()[:length]
        rest = source[length:]
        return (
            pd.concat([sma, rest])
            .ewm(span=length, adjust=False)
            .mean()
            .dropna(axis=0)
        )
