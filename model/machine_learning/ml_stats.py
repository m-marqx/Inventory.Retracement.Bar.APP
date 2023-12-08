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

import pandas as pd

class MLStats:
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
    def __init__(self, dataframe: pd.DataFrame):
        """
        Constructor method for initializing the MLStats class.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The financial DataFrame for calculations.
        """
        self.data_frame = dataframe.astype("float32").copy()

    def calculate_drawdown(self) -> pd.DataFrame:
        """
        Calculate drawdown based on the input financial DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing drawdown results.
        """
        max_results = self.data_frame.expanding(365).max()
        self.data_frame = (max_results - self.data_frame) / max_results
        return self.data_frame.fillna(0).astype("float32")

    def calculate_return_stats(
        self,
        reset_dataframe: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            self.data_frame = self.data_frame.diff().fillna()

        rt_sum_pos = self.data_frame[self.data_frame > 0].resample("A").sum()
        rt_mean_pos = self.data_frame[self.data_frame > 0].resample("A").mean()

        rt_sum_neg = (
            abs(self.data_frame[self.data_frame < 0]).resample("A").sum()
        )

        rt_mean_neg = (
            abs(self.data_frame[self.data_frame < 0]).resample("A").mean()
        )

        if reset_dataframe:
            self.data_frame = pd.DataFrame([])

        return rt_sum_pos, rt_mean_pos, rt_sum_neg, rt_mean_neg

    def calculate_expected_return(
        self,
        reset_dataframe: bool = False
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

        win_rate = self.data_frame[self.data_frame > 0].fillna(0)
        win_rate = win_rate.where(win_rate == 0, 1).astype("bool")
        win_rate = (
            win_rate.resample("A").sum() / win_rate.resample("A").count()
        )

        rt_mean_pos = self.data_frame[self.data_frame > 0].resample("A").mean()

        rt_mean_neg = (
            abs(self.data_frame[self.data_frame < 0]).resample("A").mean()
        )

        expected_return = rt_mean_pos * win_rate - rt_mean_neg * (1 - win_rate)
        expected_return = expected_return.astype("float32")

        if reset_dataframe:
            self.data_frame = pd.DataFrame([])

        return expected_return
