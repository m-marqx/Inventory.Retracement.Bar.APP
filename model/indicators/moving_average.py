import pandas as pd

# Indicadores
class moving_average:
    def sma(self, source, length):
        sma = source.rolling(length).mean()
        return sma.dropna(axis=0)

    def ema(self, source: pd.DataFrame, length: int) -> pd.DataFrame:
        """
        Calculate the Exponential Moving Average (EMA) of the input time series data.
        
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
            pd.concat([sma, rest]).ewm(span=length, adjust=False).mean().dropna(axis=0)
        )
