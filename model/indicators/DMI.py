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
        source: str = None,
        high: str = None,
        low: str = None,
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
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe param must be a DataFrame")

        if not isinstance(source, str):
            raise ValueError("source param must be a string")

        if source is None:
            if "Close" in dataframe.columns:
                self.source = dataframe["Close"]
            elif "close" in dataframe.columns:
                self.source = dataframe["close"]
        else:
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

    def true_range(self) -> pd.Series:
        """
        Calculate the True Range (TR) values for the given data.

        True Range is the greatest of the following three values:
        1. The current high minus the current low.
        2. The absolute value of the current high minus the previous close.
        3. The absolute value of the current low minus the previous close.

        Returns:
        --------
        pd.Series
            The True Range (TR) values.
        """
        true_range = np.maximum(
            self.high - self.low,
            abs(self.high - self.source.shift()),
            abs(self.low - self.source.shift())
        )
        return true_range

    def adx(
        self,
        adx_smoothing: int = 14,
        di_length: int = 14
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate the Average Directional Index (ADX) and related
        directional movement values.

        Parameters:
        -----------
        adx_smoothing : int, optional
            The smoothing period for calculating the ADX.
            (default: 14)
        di_length : int, optional
            The length of the directional movement indicator (DI) period.
            (default: 14)

        Returns:
        --------
        tuple[pd.Series, pd.Series, pd.Series]
            A tuple containing the ADX, Positive Directional Movement
            (+DI), and Negative Directional Movement (-DI) values.
        """
        trur = ma.rma(self.true_range().dropna(), di_length)

        up = self.high.diff().dropna()
        down = -self.low.diff().dropna()

        plusDM = up.where((up > down) & (up > 0), 0)
        minusDM = down.where((down > up) & (down > 0), 0)

        plus = 100 * ma.rma(plusDM, di_length) / trur
        minus = 100 * ma.rma(minusDM, di_length) / trur

        sumDM = plus + minus
        subDM = abs(plus - minus)

        adx = 100 * ma.rma(subDM / sumDM.where(sumDM != 0, 1), adx_smoothing)
        return adx, plus, minus

    def di_delta(
        self,
        adx_smoothing=14,
        di_length=14,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate the difference between the Positive Directional
        Movement (+DI) and Negative Directional Movement (-DI),
        and the ratio of +DI to -DI.

        Parameters:
        -----------
        adx_smoothing : int, optional
            The smoothing period for calculating the ADX.
            (default: 14)
        di_length : int, optional
            The length of the directional movement indicator (DI) period.
            (default: 14)

        Returns:
        --------
        tuple[pd.Series, pd.Series]
            A tuple containing the difference between +DI and -DI and
            the ratio of +DI to -DI.
        """
        _, plus, minus = self.adx(adx_smoothing, di_length)
        return plus - minus, plus / minus
