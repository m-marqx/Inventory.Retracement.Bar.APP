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
    '''The function calculates the expected value of a given trading
    data frame and returns a pandas data frame containing various
    calculated metrics related to trading.

    Parameters
    ----------
    dataframe
        a pandas DataFrame containing the trading data, including
        the result of each trade. The DataFrame
        should have a column named "Result" that contains the profit or
        loss of each trade.

    Returns
    -------
        The function `calculate_expected_value` returns a pandas
        DataFrame containing various calculated metrics related to
        trading, including gain and loss counts, mean gain and loss,
        total gain and loss, total trades, win rate, loss rate, and
        expected value (EM).

    '''
    def calculate_expected_value(self, dataframe: pd.DataFrame):
        '''This function calculates the expected value of a given
        dataframe containing trading results.

        Parameters
        ----------
        dataframe
            a pandas DataFrame containing the trading data, including
            the result of each trade. The DataFrame should have a column
            named `Result` that contains the profit or loss of each
            trade.

        Returns
        -------
            The function `calculate_expected_value` returns a pandas
            DataFrame containing various calculated metrics related
            to trading, including gain and loss counts, mean gain and
            loss, total gain and loss, total trades, win rate, loss
            rate, and expected value (EM).

        '''
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

        em_gain = data_frame["Mean_Gain"] * data_frame["Win_Rate"]
        em_loss = data_frame["Mean_Loss"] * data_frame["Loss_Rate"]
        data_frame["EM"] = em_gain - abs(em_loss)

        return data_frame


class CleanData(BaseStrategy):
    '''The function initializes an object with a copy of a given
    dataframe and a dictionary of column names, and filters the
    dataframe based on specified columns.

    Parameters
    ----------
    dataframe
        A pandas DataFrame object that contains financial data such as
        stock prices.

    '''
    def __init__(self, dataframe: pd.DataFrame):
        '''This is a constructor function that initializes an object
        with a copy of a given dataframe and a dictionary of column
        names.

        Parameters
        ----------
        dataframe
            The input parameter is a pandas DataFrame object that
            contains financial data such as stock prices.

        '''
        self.dataframe = dataframe.copy()
        self.columns = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }

    def execute(self):
        '''This function filters a dataframe and returns the filtered
        dataframe only with the OHLC columns

        Returns
        -------
            The filtered dataframe is being returned.

        '''
        try:
            self.df_filtered = self.dataframe[self.columns.values()].copy()
        except KeyError:
            self.df_filtered = self.dataframe[self.columns.keys()].copy()
            self.df_filtered.rename(columns=self.columns, inplace=True)
        return self.df_filtered


class SaveDataFrame:
    '''The function initializes an object with a dataframe attribute and
    saves the dataframe as a CSV file with specified parameters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The `dataframe` parameter is an input argument to the
        constructor of a class. It is a variable that holds a pandas
        DataFrame object, which is a two-dimensional size-mutable,
        tabular data structure with rows and columns.

    '''
    def __init__(self, dataframe: pd.DataFrame):
        '''This function initializes an object with a dataframe
        attribute and creates a directory if it does not exist.

        Parameters
        ----------
        dataframe
            The `dataframe` parameter is an input argument to the
            constructor of a class. It is a variable that holds a pandas
            DataFrame object, which is a two-dimensional size-mutable,
            tabular data structure with rows and columns.

        '''
        self.dataframe = dataframe

        self.data_path = pathlib.Path("model", "data")
        if not self.data_path.is_dir():
            self.data_path.mkdir()

    def to_csv(self, name: str) -> None:
        '''This function saves a pandas dataframe as a CSV file with
        a given name.

        Parameters
        ----------
        name : str
            The name of the CSV file that will be created. It will be
            saved with the extension ".csv".

        Returns
        -------
            a print statement indicating that the name of the saved
            CSV file.

        '''
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

    def to_parquet(self, name: str) -> None:
        '''This function saves a Pandas dataframe as a Parquet file with
        a given name.

        Parameters
        ----------
        name : str
            The name of the Parquet file that will be created. It will
            be saved with the extension ".parquet".

        Returns
        -------
            a print statement indicating that the name of the saved
            parquet file.

        '''
        str_name = f"{name}.parquet"
        dataframe_path = self.data_path.joinpath(str_name)
        self.dataframe.to_parquet(
            dataframe_path,
            index=True,
        )

        return print(str_name + " has been saved")

class CalculateTradePerformance:

    '''This is a Python class that calculates trading results based on a
    given DataFrame, capital, and trading strategy.

    Parameters
    ----------
    data_frame : pd.DataFrame
        A pandas DataFrame containing `Result` columnn.
    capital : float
        The initial capital used for trading.
    percent : bool, optional
        A boolean parameter that determines whether the results should
        be calculated as percentages or not. If set to True, the results
        will be calculated as percentages.

    '''
    def __init__(
        self,
        data_frame: pd.DataFrame,
        capital: float,
        percent: bool = False,
    ):
        '''This function initializes an object with a DataFrame,
        capital, and a boolean parameter that determines whether the
        results are expressed as percentages or not.

        Parameters
        ----------
        data_frame : pd.DataFrame
            a pandas DataFrame containing the `Result` column.
        capital : float
            The amount of capital available for trading.
        percent : bool, optional
            A boolean parameter that determines whether the result is
            expressed as a percentage or an absolute value.

        '''
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

    def calculate_results(self, gain: float, loss: float, reverse_results: bool = False):
        '''This function calculates results based on gain and loss
        conditions and can reverse the results if specified.

        Parameters
        ----------
        gain : float
            The amount of gain to be added to the `Result` column of the
            data frame if the "gain_condition" is True.
        loss : float
            The amount of loss to be added to the `Result` column of the
            data frame if the "gain_condition" is True.
        reverse_results : bool, optional
            A boolean parameter that determines whether the results
            should be reversed or not.

        '''
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
        '''This function updates the results of a trading strategy based
        on gains and losses, using either a sum or product method, and
        returns the updated results.

        Parameters
        ----------
        gain : float | pd.Series
            A float or a pandas Series representing the gains made in
            each trade or period.
        loss : float | pd.Series
            A float or a pandas Series representing the loss made in
            each trade or period.
        method : str
            The method parameter specifies whether to calculate the
            cumulative result using the `cumsum` or `cumprod` method.
        reverse_results : bool
            A boolean parameter that determines whether the results
            should be reversed or not. If set to True, the results
            will be reversed.

        Returns
        -------
            The function `update_results` is returning the updated
            instance of the class object.

        '''

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
        '''This function updates the results of a data frame based on
        fixed gain and loss values.

        Parameters
        ----------
        gain : float
            The gain parameter is a float value representing the amount
            of profit in a trade or investment.
        loss : float
            The loss parameter is a float value representing the amount
            of loss in a trade or investment.

        Returns
        -------
            a pandas DataFrame.

        '''
        if self.percent:
            gain = gain / 100 + 1
            loss = loss / 100 + 1
            self.update_results(gain, loss, "prod", False)
        else:
            self.update_results(gain, loss, "sum", False)
        return self.data_frame

    def normal(self, qty: float, coin_margined: bool) -> pd.DataFrame:
        '''The function calculates gains and losses based on input
        parameters and updates the results in a pandas dataframe.

        Parameters
        ----------
        qty : float
            The quantity of the asset being traded.
        coin_margined : bool
            A boolean value indicating whether the trade is
            coin-margined or not.

        Returns
        -------
            The function `normal` returns a pandas DataFrame.

        '''
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
