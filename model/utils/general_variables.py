from typing import Literal
from math import pi

import numpy as np
import pandas as pd
import model.utils.custom_pandas_methods


class ExternalVariables:
    """
    A class for calculating physics variables based on external data.11

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing the data.
    source_column : str, optional
        The name of the column representing the price values
        (default is "Result").

    Attributes
    ----------
    dataframe : pandas.DataFrame
        The copy of the input dataframe.
    source_column : str
        The name of the column representing the price values.

    Methods
    -------
    physics_variables(periods)
        Calculates various physics variables based on the input data.

    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        source_column: str = "Result",
        feat_last_column: str = "Signal",
    ) -> None:
        """
        Initialize the ExternalVariables class.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input dataframe containing the data.
        return_column : str, optional
            The name of the column representing the return values
            (default: "Result").

        """
        self.dataframe = dataframe.copy()
        self.source_column = source_column
        self.feat_last_column = feat_last_column

    def rolling_ratio(
        self,
        fast_length: int,
        slow_length: int,
        method: str,
    ) -> pd.DataFrame:
        """
        Calculate a rolling ratio of two rolling averages.

        This method computes a rolling ratio using two rolling averages
        based on specified parameters.

        Parameters:
        -----------
        fast_length : int
            The window size for the fast rolling average.
        slow_length : int
            The window size for the slow rolling average.
        method : str
            The method used for rolling averages
            (e.g., 'mean', 'std', 'sum').

        Returns:
        --------
        pd.DataFrame
            The original DataFrame with an additional column for the
            rolling ratio.

        Raises:
        -------
        ValueError
            If an invalid method is specified.
        """
        if fast_length == slow_length:
            raise ValueError("fast_length and slow_length must be different")

        fast_rolling = self.dataframe[self.source_column].rolling(fast_length)
        slow_rolling = self.dataframe[self.source_column].rolling(slow_length)

        try:
            fast_rolling = getattr(fast_rolling, method)()
            slow_rolling = getattr(slow_rolling, method)()
        except AttributeError as exc:
            raise ValueError(f"Invalid method '{method}'") from exc

        self.dataframe["rolling_std_ratio"] = fast_rolling / slow_rolling

        return self.dataframe

    def ratio_variables(self, column: str, length: int) -> pd.DataFrame:
        """
        Compute ratio-based variables for a given column.

        Parameters:
        -----------
        column : str
            The name of the column for which ratios will be calculated.
        length : int
            The window length for rolling statistics.

        Returns:
        --------
        pd.DataFrame
            Returns the DataFrame with ratio-based variables added.
        """
        ma_feat = "MA_" + column
        std_feat = "STD_" + column

        self.dataframe[ma_feat] = (
            self.dataframe[column]
            .rolling(window=length)
            .mean()
        )

        self.dataframe[std_feat] = (
            self.dataframe[column]
            .rolling(window=length)
            .std()
        )

        self.dataframe[ma_feat + "_ratio"] = (
            self.dataframe[column]
            / self.dataframe[ma_feat] - 1
        )

        self.dataframe[std_feat + "_ratio"] = (
            self.dataframe[column]
            / self.dataframe[std_feat] - 1
        )

        self.dataframe = self.dataframe.reorder_columns(
            self.feat_last_column, self.dataframe.columns[-4:]
        )

        return self.dataframe

    def physics_variables(self, periods: int) -> pd.DataFrame:
        """
        Calculates various physics variables based on the input data.

        Parameters
        ----------
        periods : int
            The number of periods to consider for calculations.

        Returns
        -------
        pandas.DataFrame
            The modified dataframe with added physics variables.

        """

        self.dataframe["velocity"] = (
            self.dataframe[self.source_column]
            - self.dataframe[self.source_column].shift(periods)
        ) / periods

        self.dataframe["acceleration"] = (
            self.dataframe["velocity"]
            - self.dataframe["velocity"].shift(periods)
        ) / periods

        self.dataframe["mass"] = (
            self.dataframe[self.source_column].rolling(periods).sum()
        )

        self.dataframe["force"] = (
            self.dataframe["mass"] * self.dataframe["acceleration"]
        )

        self.dataframe["kinetic"] = (
            0.5
            * self.dataframe["mass"]
            * self.dataframe["velocity"]
            * self.dataframe["velocity"]
        )

        opposite_leg = (
            self.dataframe["mass"]
            - self.dataframe[self.source_column].rolling(1).sum()
        )
        adjacent_leg = periods

        self.dataframe["tangent"] = opposite_leg / adjacent_leg

        self.dataframe = self.dataframe.dropna(axis=0)

        self.dataframe["work"] = self.dataframe["force"] * np.cos(
            np.arctan(self.dataframe["tangent"])
        )

        self.dataframe["potential_energy"] = (
            opposite_leg
            * self.dataframe["mass"]
        )

        self.dataframe["torque"] = self.dataframe["force"] * np.sin(
            np.arctan(self.dataframe["tangent"])
        )

        self.dataframe["momentum"] = (
            self.dataframe["mass"]
            * self.dataframe["velocity"]
        )

        self.dataframe["gravity"] = self.dataframe["mass"] ** 2

        self.dataframe.dropna(axis=0, inplace=True)

        self.dataframe["velocity"] = pd.qcut(
            self.dataframe["velocity"], periods, labels=False
        )
        self.dataframe["acceleration"] = pd.qcut(
            self.dataframe["acceleration"], periods, labels=False
        )

        self.dataframe["mass"] = pd.qcut(
            self.dataframe["mass"],
            periods,
            labels=False,
        )

        self.dataframe["force"] = pd.qcut(
            self.dataframe["force"],
            periods,
            labels=False,
        )

        self.dataframe["kinetic"] = pd.qcut(
            self.dataframe["kinetic"],
            periods,
            labels=False,
        )

        self.dataframe["work"] = pd.qcut(
            self.dataframe["work"],
            periods,
            labels=False,
        )

        self.dataframe["potential_energy"] = pd.qcut(
            self.dataframe["potential_energy"],
            periods,
            labels=False,
        )

        self.dataframe["torque"] = pd.qcut(
            self.dataframe["torque"],
            periods,
            labels=False,
        )

        self.dataframe["momentum"] = pd.qcut(
            self.dataframe["momentum"],
            periods,
            labels=False,
        )

        self.dataframe["gravity"] = pd.qcut(
            self.dataframe["gravity"],
            periods,
            labels=False,
        )

        self.dataframe["tangent"] = pd.qcut(
            self.dataframe["tangent"],
            periods,
            labels=False,
        )

        self.dataframe = self.dataframe.reorder_columns(
            self.feat_last_column, self.dataframe.columns[-11:]
        )
        return self.dataframe

    def schumann_resonance(
        self,
        source_column: str,
        n_length: int,
        method: Literal[
            "equatorial",
            "polar",
            "mean",
            "custom"] = "equatorial",
        circuference: float | None = None
    ) -> pd.Series:
        """
        Calculate Schumann resonance frequencies based on a source
        column.

        This method calculates Schumann resonance frequencies based
        on the values in a source column. The Schumann resonance
        frequencies are influenced by the method used and Earth's
        circumference.

        Parameters:
        -----------
        source_column : str
            The name of the source column containing integer values.
        n_length : int
            The desired length of the resulting values.
        method : Literal["equatorial", "polar", "mean", "custom"],
        optional
            The method to calculate Schumann resonance frequencies:
            - 'equatorial': For equatorial method
            - 'polar': For polar method.
            - 'mean': For mean Earth circumference method.
            - 'custom': For a custom method using 'circuference'.
            (default: "equatorial")
        circuference : float, optional
            The custom circumference value
            (required if 'method' is 'custom').
            (default: None)

        Returns:
        --------
        pd.Series
            A pandas Series containing the calculated Schumann
            resonance frequencies.

        Raises:
        -------
        ValueError:
            If 'method' is 'custom' and 'circumference' is not specified.
        """
        light_speed_km = 299792.458

        if method != 'custom':
            circuferences_km = {
                "equatorial" : 2 * pi * 6378.137,
                "polar" : 2 * pi * 6356.752,
                "mean" : 2 * pi * 6371.0,
            }
            earth_circuferences_km = pd.Series(circuferences_km)
            sr_constant = light_speed_km / earth_circuferences_km
        else:
            if circuference is None:
                raise ValueError("circumference parameter must be specified")
            sr_constant = light_speed_km / circuference

        formula_constant = (
            sr_constant[method] if method != "custom"
            else sr_constant
        )

        values = self.dataframe[source_column].astype("int64").copy()
        n_digits = values.astype("str").str.len().astype("float64")
        converted_values = values * 10 ** (n_length - n_digits)
        schumann_resonance = (
            formula_constant
            * np.sqrt((converted_values * (converted_values + 1)))
        )
        return schumann_resonance

    def fox_trap(self) -> pd.DataFrame:
        """
        Identify 'Fox Trap' conditions in financial data.

        Applies the 'Fox Trap' trading strategy by evaluating conditions
        against moving averages and price action in a DataFrame that
        must contain high, low, and close prices.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'long' and 'short' 'Fox Trap' signals, named
            by the moving average column provided.

        Raises
        ------
        ValueError
            If 'high', 'low', or 'close' prices are absent in the
            DataFrame.
        """
        required_columns = ['High', 'Low', 'Close']

        column_mapping = {}
        for col in required_columns:
            found_columns = [
                column
                for column in self.dataframe.columns
                if column == col or column.capitalize() == col
            ]

            if not found_columns:
                raise ValueError(
                    f"Column '{col}' not found in dataframe with either"
                    " lowercase or capitalized format"
                )
            column_mapping[col] = found_columns[0]

        high = self.dataframe[column_mapping['high']]
        low = self.dataframe[column_mapping['low']]
        close = self.dataframe[column_mapping['close']]

        moving_average = self.dataframe[self.source_column]
        high_fox_trap_condition = (
            (close > moving_average)
            & (low < moving_average)
        )

        low_fox_trap_condition = (
            (close < moving_average)
            & (high > moving_average)
        )

        shifted_high = high.shift()
        shifted_low = low.shift()

        buy_high_fox_trap_condition = (
            high_fox_trap_condition.shift()
            & (high > shifted_high)
        )

        sell_low_fox_trap_condition = (
            low_fox_trap_condition.shift()
            & (low < shifted_low)
        )

        high_column = f"fox_trap_{self.source_column}_long"
        low_column = f"fox_trap_{self.source_column}_short"
        hl_columns = [high_column,low_column]

        self.dataframe[high_column] = buy_high_fox_trap_condition
        self.dataframe[low_column] = sell_low_fox_trap_condition
        self.dataframe[hl_columns] = self.dataframe[hl_columns].astype('int8')

        return self.dataframe
