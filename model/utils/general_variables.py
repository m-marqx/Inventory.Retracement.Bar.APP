from typing import Literal
from math import pi

import numpy as np
import pandas as pd
import model.utils.custom_pandas_methods


class ExternalVariables:
    """
    A class for calculating variables based on the input data.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing the data.
    source_column : str, optional
        The name of the column representing the price values
        (default: "Result")
    feat_last_column : str, optional
        The name of the column representing the last feature
        (default: "Signal")
    return_type : Literal["short", "full"], optional
        The return type of methods ('short' returns only calculated
        values, 'full' returns the modified DataFrame with added
        columns).
        (default: "short")

    Attributes
    ----------
    dataframe : pandas.DataFrame
        The copy of the input dataframe.
    source_column : str
        The name of the column representing the price values.
    feat_last_column : str
        The name of the column representing the last feature.
    return_type : Literal["short", "full"]
        The return type of methods.

    Methods
    -------
    rolling_ratio(fast_length, slow_length, method)
        Calculate a rolling ratio of two rolling averages.

    ratio_variable(length, method)
        Compute ratio-based variables.

    physics_variables(periods)
        Calculates various physics variables based on the input data.

    schumann_resonance(source_column,
    n_length, method, circuference=None)
        Calculate Schumann resonance frequencies based on a source
        column.

    fox_trap()
        Identify 'Fox Trap' conditions in on the input data.

    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        source_column: str = "Result",
        feat_last_column: str = "Signal",
        return_type: Literal["short", "full"] = "short",
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
        self.return_type = return_type

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
            (e.g., 'mean', 'std', 'median').

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

        rolling_std_ratio = fast_rolling / slow_rolling

        if self.return_type == "short":
            return rolling_std_ratio

        self.dataframe["rolling_std_ratio"] = rolling_std_ratio

        return self.dataframe

    def ratio_variable(self, length: int, method: str) -> pd.DataFrame:
        """
        Compute ratio-based variables.

        Parameters:
        -----------
        length : int
            The window length for rolling statistics.
        method : str
            The method used for rolling averages
            (e.g., 'mean', 'std', 'median').
        Returns:
        --------
        pd.DataFrame
            Returns the DataFrame with ratio-based variables added.
        """
        rolling_data = self.dataframe[self.source_column].rolling(length)
        rolling_data = getattr(rolling_data, method)()

        ratio_variable = (
            self.dataframe[self.source_column]
            / rolling_data - 1
        )

        if self.return_type == "short":
            return ratio_variable.iloc[:, -1:]

        self.dataframe["ratio_variable"] = ratio_variable


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

        if self.return_type == "short":
            return self.dataframe.iloc[:, -11:]

        self.dataframe = self.dataframe.reorder_columns(
            self.feat_last_column, self.dataframe.columns[-11:]
        )
        return self.dataframe

    def schumann_resonance(
        self,
        n_length: int,
        method: Literal[
            "equatorial",
            "polar",
            "mean",
            "custom"] = "equatorial",
        circumference : float | None = None
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

        if method not in ["equatorial", "polar", "mean", "custom"]:
            raise ValueError(
                f"Invalid method '{method}'. "
                "Valid methods are"
                " 'equatorial', 'polar', 'mean', and 'custom'."
            )

        match method:
            case "equatorial":
                circumference = 2 * pi * 6378.137
            case "mean":
                circumference = 2 * pi * 6371.0
            case "polar":
                circumference = 2 * pi * 6356.752
            case "custom":
                if not circumference:
                    raise ValueError(
                        "when method is custom, the circumference parameter"
                        " must be specified"
                    )

        sr_constant = light_speed_km / circumference

        n_digits = (
            self.dataframe[self.source_column].astype("str")
            .str.replace(".","")
        )

        expoent = (n_length - 1) - n_digits.str.len().astype("float64")
        converted_values = n_digits.astype("float64") * 10 ** expoent

        schumann_resonance = (
            sr_constant
            * np.sqrt(converted_values * (converted_values + 1))
        )
        if self.return_type == "short":
            return schumann_resonance

        self.dataframe["schumann_resonance"] = schumann_resonance

        return self.dataframe

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

        if self.return_type == "short":
            return pd.concat([
                buy_high_fox_trap_condition,
                sell_low_fox_trap_condition
            ], axis=1).astype('int8')

        self.dataframe[high_column] = buy_high_fox_trap_condition
        self.dataframe[low_column] = sell_low_fox_trap_condition
        self.dataframe[hl_columns] = self.dataframe[hl_columns].astype('int8')

        return self.dataframe
