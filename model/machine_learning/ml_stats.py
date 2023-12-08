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
