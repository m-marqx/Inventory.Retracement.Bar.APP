from abc import ABC, abstractmethod
import pathlib
import numpy as np
import pandas as pd


class BaseStrategy(ABC):

    '''This is a Python class with an abstract method `execute()` that
    raises a `NotImplementedError`, and an `__init__()` method that
    takes a pandas DataFrame as an argument and assigns it to an
    instance variable `df_filtered`.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A pandas DataFrame object that will be used as input for
        the class.

    '''
    def __init__(self, dataframe: pd.DataFrame):
        '''This is a constructor function that initializes an instance
        variable "df_filtered" with a pandas DataFrame passed as an
        argument.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The parameter "dataframe" is a pandas DataFrame object that
            is passed as an argument to the constructor of a class. The
            constructor initializes an instance variable "df_filtered"
            with the value of the passed DataFrame object.

        '''
        self.df_filtered = dataframe

    @abstractmethod
    def execute(self):
        '''The function "execute" is defined but raises a
        NotImplementedError.

        '''
        raise NotImplementedError


class BrokerEmulator:
    '''The function initializes an object with a pandas DataFrame and
    calculates the result of a broker emulator based on the high and low
    prices of the DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The parameter "dataframe" is a pandas DataFrame object that is
        passed as an argument to the constructor of a class. The
        constructor initializes an instance variable "self.dataframe"
        with the value of the passed DataFrame object. This allows the
        DataFrame to be accessed and manipulated within the class
        methods.
    '''
    def __init__(self, dataframe: pd.DataFrame):
        '''This is a constructor function that initializes an object
        with a pandas DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The parameter "dataframe" is a pandas DataFrame object that
            is passed as an argument to the
            constructor of a class. The constructor initializes an
            instance variable "self.dataframe" with the value of the
            passed DataFrame object. This allows the DataFrame to be
            accessed and manipulated within the class methods.
        '''
        self.dataframe = dataframe

    def broker_emulator_result(self):
        '''This method calculates the result of a broker emulator
        based on the high and low prices of a given dataframe.

        Returns
        -------
            The method `broker_emulator_result` returns the instance
            of the class that it belongs to (`self`).

        '''
        self.distance_high_to_open = self.dataframe["high"] - self.dataframe["open"]
        self.distance_low_to_open = self.dataframe["open"] - self.dataframe["low"]
        self.broker_emulator = np.where(
            self.distance_high_to_open < self.distance_low_to_open,
            self.dataframe["high"],
            self.dataframe["low"],
        )

        self.dataframe["order_fill_price"] = self.broker_emulator
        self.sell_prices = self.dataframe[["Take_Profit", "Stop_Loss"]]

        self.sell_diffs = np.abs(
            self.sell_prices - self.dataframe["order_fill_price"].values[:, np.newaxis]
        )
        self.duplicate = self.dataframe["Signal"] == -2

        self.TP_is_close = self.sell_diffs["Take_Profit"] < self.sell_diffs["Stop_Loss"]
        self.profit = self.dataframe["Take_Profit"] - self.dataframe["Entry_Price"]
        self.loss = self.dataframe["Stop_Loss"] - self.dataframe["Entry_Price"]
        self.dataframe["Result"] = np.where(
            self.duplicate & self.TP_is_close, self.profit, self.dataframe["Result"]
        )
        self.dataframe["Result"] = np.where(
            self.duplicate & ~self.TP_is_close, self.loss, self.dataframe["Result"]
        )

        return self

    def exit_price(self):
        '''This function calculates the exit price for a trading
        strategy based on the take profit and stop loss levels.

        Returns
        -------
            a pandas Series object containing the exit prices for each
            row in the input dataframe.
        '''
        self.broker_emulator_result()

        self.data_frame = self.dataframe.copy()
        self.data_frame["Exit_Price"] = np.nan

        self.data_frame["Exit_Price"] = np.where(
            (self.data_frame["high"] > self.data_frame["Take_Profit"])
            & self.data_frame["Close_Position"],
            self.data_frame["Take_Profit"],
            self.data_frame["Exit_Price"],
        )

        self.data_frame["Exit_Price"] = np.where(
            (self.data_frame["low"] < self.data_frame["Stop_Loss"])
            & self.data_frame["Close_Position"],
            self.data_frame["Stop_Loss"],
            self.data_frame["Exit_Price"],
        )

        self.data_frame["Exit_Price"] = np.where(
            (self.data_frame["high"] > self.data_frame["Take_Profit"])
            & self.data_frame["Close_Position"],
            self.data_frame["Take_Profit"],
            self.data_frame["Exit_Price"],
        )

        self.data_frame["Exit_Price"] = np.where(
            self.duplicate & self.TP_is_close,
            self.data_frame["Take_Profit"],
            self.data_frame["Exit_Price"],
        )

        self.data_frame["Exit_Price"] = np.where(
            self.duplicate & ~self.TP_is_close,
            self.data_frame["Stop_Loss"],
            self.data_frame["Exit_Price"],
        )

        return self.data_frame["Exit_Price"]

class DataProcess:

    '''The code defines a class with methods to classify and filter a
    transposed pandas DataFrame.

    Parameters
    ----------
    data_frame : pd.DataFrame
        A pandas DataFrame that will be transposed and used for
        classification and filtering.
    min_value : float
        The minimum value that a row must have in the last column to be
        included in the "best_positive_results" property.

    '''
    def __init__(self, data_frame: pd.DataFrame, min_value: float = 0.0):
        '''This is a constructor function that initializes an object
        with a transposed copy of a pandas DataFrame, the name of the
        last column, and a minimum value.

        Parameters
        ----------
        data_frame : pd.DataFrame
            A pandas DataFrame that will be transposed and stored as an
            attribute of the class.
        min_value : float
            A float value that represents the minimum value allowed for
            a certain operation or calculation. It is an optional
            parameter with a default value of 0.0.

        '''
        self.df_transposed = data_frame.copy().T
        self.last_column_name = self.df_transposed.columns[-1]
        self.min_value = min_value

    def classify_dataframe(self, index: bool = False):
        '''This function classifies a transposed dataframe by ranking
        its last column and returning a melted version sorted by rank
        and index.

        Parameters
        ----------
        index : bool, optional
            The "index" parameter is a boolean flag that indicates
            whether or not to include the original index of the
            DataFrame in the resulting melted DataFrame. If "index" is
            set to True, the original index will be included as a column
            in the melted DataFrame.

        Returns
        -------
            a melted and sorted pandas DataFrame with a new column
            "rank" added to the original transposed DataFrame. The
            melted DataFrame has three columns: "rank", "index",
            and "result". The DataFrame is sorted by "rank" and "index".

        '''
        if not index:
            self.df_transposed = self.df_transposed.reset_index(drop=True).copy().T

        self.df_transposed["rank"] = (
            self.df_transposed[self.last_column_name].rank(method="min") - 1
        )
        return pd.melt(
            self.df_transposed,
            id_vars=["rank"],
            var_name="index",
            value_name="result",
        ).sort_values(by=["rank", "index"])

    @property
    def best_positive_results(self):
        '''This function returns a transposed dataframe with rows
        filtered by a minimum value and sorted by the last column in
        descending order.

        Returns
        -------
            The code is returning a transposed DataFrame containing the
            rows with positive values greater than the minimum value
            specified by the user, sorted in descending order based on
            the last column of the DataFrame. The returned DataFrame is
            filtered and sorted based on the last column of the original
            DataFrame.

        '''
        df_transposed_last_column = self.df_transposed.iloc[:, [-1]]

        filtered_df = df_transposed_last_column[df_transposed_last_column > self.min_value]
        filtered_df.dropna(inplace=True)

        filtered_df_sorted = filtered_df.sort_values(
            by=str(filtered_df.columns[-1]),
            ascending=False,
        ).index

        return self.df_transposed.loc[filtered_df_sorted].T


class Math:
    def calculate_expected_value(self, dataframe):
        data_frame = dataframe.query("Result != 0")[["Result"]].copy()

        gain = data_frame["Result"] > 0
        loss = data_frame["Result"] < 0

        data_frame["Gain_Count"] = np.where(gain, 1, 0)
        data_frame["Loss_Count"] = np.where(loss, 1, 0)

        data_frame["Gain_Count"] = data_frame["Gain_Count"].cumsum()
        data_frame["Loss_Count"] = data_frame["Loss_Count"].cumsum()

        query_gains = data_frame.query("Result > 0")["Result"]
        query_loss = data_frame.query("Result < 0")["Result"]

        data_frame["Mean_Gain"] = query_gains.expanding().mean()
        data_frame["Mean_Loss"] = query_loss.expanding().mean()

        data_frame["Mean_Gain"].fillna(method="ffill", inplace=True)
        data_frame["Mean_Loss"].fillna(method="ffill", inplace=True)

        data_frame["Total_Gain"] = (
            np.where(gain, data_frame["Result"], 0)
            .cumsum()
        )

        data_frame["Total_Loss"] = (
            np.where(loss, data_frame["Result"], 0)
            .cumsum()
        )

        total_trade = data_frame["Gain_Count"] + data_frame["Loss_Count"]
        win_rate = data_frame["Gain_Count"] / total_trade
        loss_rate = data_frame["Loss_Count"] / total_trade

        data_frame["Total_Trade"] = total_trade
        data_frame["Win_Rate"] = win_rate
        data_frame["Loss_Rate"] = loss_rate

        # expected self.mathematical calculation
        em_gain = data_frame["Mean_Gain"] * data_frame["Win_Rate"]
        em_loss = data_frame["Mean_Loss"] * data_frame["Loss_Rate"]
        data_frame["EM"] = em_gain - abs(em_loss)

        return data_frame


class CleanData(BaseStrategy):
    #! Don't convert the values to float32
    #! because it significantly reduces the precision of the data.
    def __init__(self, dataframe):
        self.dataframe = dataframe.copy()
        self.columns = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }

    def execute(self):
        try:
            self.df_filtered = self.dataframe[self.columns.values()].copy()
        except KeyError:
            self.df_filtered = self.dataframe[self.columns.keys()].copy()
            self.df_filtered.rename(columns=self.columns, inplace=True)
        return self.df_filtered


class SaveDataFrame:
    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.data_path = pathlib.Path("model", "data")
        if not self.data_path.is_dir():
            self.data_path.mkdir()

    def to_csv(self, name) -> None:
        str_name = f"{name}.csv"
        dataframe_path = self.data_path.joinpath(str_name)
        columns = self.dataframe.columns
        self.dataframe.to_csv(
            dataframe_path,
            index=True,
            header=columns,
            sep=";",
            decimal=".",
            encoding="utf-8",
        )

        return print(str_name + " has been saved")

    def to_parquet(self, name) -> None:
        str_name = f"{name}.parquet"
        dataframe_path = self.data_path.joinpath(str_name)
        self.dataframe.to_parquet(
            dataframe_path,
            index=True,
        )

        return print(str_name + " has been saved")

class CalculateTradePerformance:
    def __init__(
        self,
        data_frame: pd.DataFrame,
        capital: float,
        percent: bool = False,
    ):
        self.data_frame = data_frame.copy()
        self.gain_condition = self.data_frame["Result"] > 0
        self.loss_condition = self.data_frame["Result"] < 0
        self.close_trade = self.data_frame["Result"] != 0
        self.capital = capital
        self.data_frame["Capital"] = self.capital
        self.data_frame["Result"] = np.where(
            ~self.close_trade,
            np.nan,
            self.data_frame["Result"],
        )
        self.percent = percent

    def calculate_results(self, gain, loss, reverse_results: bool = False):
        if reverse_results:
            self.data_frame["Result"] = np.where(
                self.gain_condition,
                -gain,
                self.data_frame["Result"],
            )
            self.data_frame["Result"] = np.where(
                self.loss_condition,
                -loss,
                self.data_frame["Result"],
            )
            self.data_frame["Result"] = self.data_frame["Result"] + 2

        else:
            self.data_frame["Result"] = np.where(
                self.gain_condition,
                gain,
                self.data_frame["Result"],
            )
            self.data_frame["Result"] = np.where(
                self.loss_condition,
                loss,
                self.data_frame["Result"],
            )

    def update_results(self, gain: float | pd.Series, loss: float | pd.Series, method: str, reverse_results: bool):

        self.calculate_results(gain, loss, reverse_results)

        if method == "sum":
            self.data_frame["Cumulative_Result"] = (
                self.data_frame["Result"]
                .cumsum()
                .ffill()
            )
            self.data_frame["Capital"] = (
                self.data_frame["Cumulative_Result"]
                + self.data_frame["Capital"]
            )
        elif method == "prod":
            self.data_frame["Cumulative_Result"] = (
                self.data_frame["Result"]
                .cumprod()
                .ffill()
            )
            self.data_frame["Capital"] = (
                self.data_frame["Cumulative_Result"]
                * self.data_frame["Capital"]
            )

        if self.percent:
            self.data_frame["Result"] = (self.data_frame["Result"] - 1) * 100
            self.data_frame["Cumulative_Result"] = (
                self.data_frame["Cumulative_Result"] - 1
            ) * 100

        self.data_frame["Cumulative_Result"].fillna(0, inplace=True)
        self.data_frame["Capital"].fillna(self.capital, inplace=True)

        return self

    def fixed(self, gain: float, loss: float) -> pd.DataFrame:
        if self.percent:
            gain = gain / 100 + 1
            loss = loss / 100 + 1
            self.update_results(gain, loss, "prod", False)
        else:
            self.update_results(gain, loss, "sum", False)
        return self.data_frame

    def normal(self, qty: float, coin_margined: bool) -> pd.DataFrame:
        if self.percent:
            if coin_margined:
                gain = (
                    self.data_frame["Entry_Price"]
                    / self.data_frame["Take_Profit"]
                    * qty
                )
                loss = (
                    self.data_frame["Entry_Price"]
                    / self.data_frame["Stop_Loss"]
                    * qty
                )
            else:
                gain = (
                    self.data_frame["Take_Profit"]
                    / self.data_frame["Entry_Price"]
                    * qty
                )
                loss = (
                    self.data_frame["Stop_Loss"]
                    / self.data_frame["Entry_Price"]
                    * qty
                )
            method = "prod"
        else:
            gain = self.data_frame["Result"] * qty
            loss = self.data_frame["Result"] * qty
            method = "sum"
        self.update_results(gain, loss, method, coin_margined)
        return self.data_frame
