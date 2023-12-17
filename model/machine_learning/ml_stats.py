"""
Class: MLStats

A class for calculating drawdown, return statistics, and expected return
based on a given financial DataFrame.

Methods:
--------
- __init__(self, dataframe: pd.DataFrame): Constructor method for \
    initializing the MLStats class.
- calculate_drawdown(self) -> pd.DataFrame: Calculate drawdown \
    based on the input financial DataFrame.
- calculate_return_stats(self, reset_dataframe: bool = False) \
    -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: \
        Calculate return statistics.
- calculate_expected_return(self, reset_dataframe: bool = False) \
    -> pd.DataFrame: Calculate expected return.

Attributes:
-----------
- data_frame: pd.DataFrame
    The financial DataFrame used for calculations.
"""

from typing import Literal
import pandas as pd

class ModelMetrics:
    """
    A class for calculating drawdown, return statistics, and expected
    return based on a given financial DataFrame.

    Methods:
    --------
    - __init__(self, dataframe: pd.DataFrame): Constructor method
    for initializing the MLStats class.
    - calculate_drawdown(self) -> pd.DataFrame: Calculate drawdown
    based on the input financial DataFrame.
    - calculate_return_stats(self, reset_dataframe: bool = False) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: \
            Calculate return statistics.
    - calculate_expected_return(self, reset_dataframe: bool = False) \
        -> pd.DataFrame: Calculate expected return.

    Attributes:
    -----------
    - data_frame: pd.DataFrame
        The financial DataFrame used for calculations.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        period: str | int = "W"
    ):
        """
        Constructor method for initializing the MLStats class.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The financial DataFrame for calculations.
        """
        self.data_frame = dataframe.copy()
        self.period = period
        self.is_int_period = isinstance(period, int)

    def calculate_drawdown(self) -> pd.DataFrame:
        """
        Calculate drawdown based on the input financial DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing drawdown results.
        """
        max_results = self.data_frame.expanding(365).max()
        drawdown = (max_results - self.data_frame) / max_results
        return drawdown.fillna(0)

    def calculate_mean_drawdown(self) -> pd.DataFrame:
        """
        Calculate mean drawdown based on the input financial DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing mean drawdown results.
        """
        drawdown = self.calculate_drawdown()

        if self.is_int_period:
            drawdown = (
                drawdown.fillna(0).astype('float32')
                .resample(self.period).mean()
            )

        drawdown = (
            drawdown.fillna(0).astype('float32')
            .rolling(self.period).mean()
        )

    def calculate_payoff(
        self,
        method: Literal['sum', 'mean'] = 'sum',
    ) -> pd.DataFrame:
        """
        Calculate return statistics.

        Parameters:
        -----------
        reset_dataframe : bool, optional
            Flag to reset the internal DataFrame
            (default: False).

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple containing positive sum, positive mean, negative sum,
            and negative mean return statistics.
        """
        if self.data_frame.min() > 0:
            rt = self.data_frame.copy().diff().fillna(0)
        else:
            rt = self.data_frame.copy()

        pos_values = rt[rt > 0]
        neg_values = abs(rt[rt < 0])

        if method == 'sum':
            pos_values = (
                pos_values.fillna(0).rolling(self.period).sum()
                if self.is_int_period
                else pos_values.ffill().rolling(self.period).sum().fillna(0)
            )

            neg_values = (
                neg_values.fillna(0).rolling(self.period).sum()
                if self.is_int_period
                else neg_values.ffill().rolling(self.period).sum().fillna(0)
            )

        else:
            pos_values = (
                pos_values.ffill().rolling(self.period).mean().fillna(0)
                if self.is_int_period
                else pos_values.ffill().rolling(self.period).mean().fillna(0)
            )
            neg_values = (
                neg_values.ffill().rolling(self.period).mean().fillna(0)
                if self.is_int_period
                else neg_values.ffill().rolling(self.period).mean().fillna(0)
            )

        payoff = pos_values / neg_values
        return payoff.dropna()

    def calculate_expected_return(self):
        """
        Calculate expected return.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing expected return values.
        """
        if self.is_int_period:
            return self.__resample_calculate_expected_return()
        return self.__rolling_calculate_expected_return()

    def __resample_calculate_expected_return(
        self,
    ) -> pd.DataFrame:
        """
        Calculate expected return.

        Parameters:
        -----------
        reset_dataframe : bool, optional
            Flag to reset the internal DataFrame
            (default: False).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing expected return values.
        """
        if self.data_frame.min() > 0:
            self.data_frame = self.data_frame.diff().fillna()

        rt = self.data_frame.to_frame().diff().fillna(0)

        win_rate = rt[rt > 0].fillna(0)
        win_rate = win_rate.where(win_rate == 0, 1).astype('bool')
        win_rate = (
            win_rate.resample(self.period).sum()
            / win_rate.resample(self.period).count()
        )

        rt_mean_pos = rt[rt > 0].resample(self.period).mean()
        rt_mean_neg = abs(rt[rt < 0]).resample(self.period).mean()

        expected_return = rt_mean_pos * win_rate - rt_mean_neg * (1 - win_rate)
        expected_return = expected_return.astype('float32')

        return expected_return.dropna()

    def __resample_calculate_win_rate(
        self,
    ) -> pd.DataFrame:
        rt = self.data_frame.copy().diff().fillna(0)

        win_rate = rt[rt > 0].fillna(0)
        win_rate = win_rate.where(win_rate == 0, 1).astype('bool')
        win_rate = (
            win_rate.resample(self.period).sum()
            / win_rate.resample(self.period).count()
        )
        return win_rate

    def __rolling_calculate_expected_return(
        self,
        period: int = 30,
    ) -> pd.DataFrame:
        rt = self.data_frame.diff().fillna(0)

        pos_values = rt[rt > 0]
        neg_values = abs(rt[rt < 0])

        pos_count = pos_values.rolling(period).count()
        neg_count = neg_values.rolling(period).count()

        win_rate = pos_count / (pos_count + neg_count)

        pos_mean = pos_values.ffill().rolling(period).mean().fillna(0)
        neg_mean = neg_values.ffill().rolling(period).mean().fillna(0)

        expected_return = pos_mean * win_rate - neg_mean * (1 - win_rate)
        expected_return = expected_return.astype('float32')

        return expected_return.dropna()

    def __rolling_calculate_win_rate(
        self,
        period: int = 30,
    ) -> pd.DataFrame:
        rt = self.data_frame.diff().fillna(0)

        pos_values = rt[rt > 0]
        neg_values = abs(rt[rt < 0])

        pos_count = pos_values.rolling(period).count()
        neg_count = neg_values.rolling(period).count()

        win_rate = pos_count / (pos_count + neg_count)
        return win_rate

    def calculate_win_rate(self):
        """
        Calculate win rate.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing win rate values.
        """
        if self.is_int_period:
            return self.__resample_calculate_win_rate()
        return self.__rolling_calculate_win_rate()
