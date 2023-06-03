import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import pathlib


class BaseStrategy(ABC):
    def __init__(self, dataframe: pd.DataFrame):
        self.df_filtered = dataframe

    @abstractmethod
    def execute(self):
        raise NotImplementedError


class BrokerEmulator:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def broker_emulator_result(self):
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
    def __init__(self, data_frame):
        self.df_transposed = data_frame.copy().T
        self.last_column_name = self.df_transposed.columns[-1]

    def classify_dataframe(self, index: bool = False):
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
