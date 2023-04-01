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
    sma(source: pd.DataFrame, length: int) -> pd.DataFrame:
        Calculate the Simple Moving Average (SMA)
        of the input time series data.

    ema(source: pd.DataFrame, length: int) -> pd.DataFrame:
        Calculate the Exponential Moving Average (EMA)
        of the input time series data.
    """
    def sma(self, source: pd.DataFrame, length: int) -> pd.DataFrame:
        """
        Calculate the Simple Moving Average (SMA) 
        of the input time series data.

        Parameters:
        -----------
        source : pd.DataFrame
            The time series data to calculate the SMA for.
        length : int
            The number of periods to include in the SMA calculation.

        Returns:
        --------
        pd.DataFrame
            The calculated SMA time series data.
        """
        sma = source.rolling(length).mean()
        return sma.dropna(axis=0)

    def ema(self, source: pd.DataFrame, length: int) -> pd.DataFrame:
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
