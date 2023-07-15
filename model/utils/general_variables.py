import numpy as np
import pandas as pd


class ExternalVariables:
    """
    A class for calculating physics variables based on external data.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing the data.
    return_column : str, optional
        The name of the column representing the return values
        (default is "Result").

    Attributes
    ----------
    dataframe : pandas.DataFrame
        The copy of the input dataframe.
    return_column : str
        The name of the column representing the return values.

    Methods
    -------
    physics_variables(periods)
        Calculates various physics variables based on the input data.

    """

    def __init__(self, dataframe: pd.DataFrame, return_column: str = "Result", feat_last_column: str = "Signal"):
        """
        Initialize the ExternalVariables class.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input dataframe containing the data.
        return_column : str, optional
            The name of the column representing the return values
            (default is "Result").

        """
        self.dataframe = dataframe.copy()
        self.return_column = return_column
        self.feat_last_column = feat_last_column

    def physics_variables(self, periods: int):
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
            self.dataframe[self.return_column]
            - self.dataframe[self.return_column].shift(periods)
        ) / periods

        self.dataframe["acceleration"] = (
            self.dataframe["velocity"]
            - self.dataframe["velocity"].shift(periods)
        ) / periods

        self.dataframe["mass"] = (
            self.dataframe[self.return_column].rolling(periods).sum()
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
            - self.dataframe[self.return_column].rolling(1).sum()
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

        self.dataframe.drop(
            ["mass", "tangent", "momentum"],
            axis=1,
            inplace=True,
        )

        self.dataframe = self.dataframe.reorder_columns(
            self.feat_last_column, self.dataframe.columns[-8:]
        )
        return self.dataframe
