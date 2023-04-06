import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, dataframe: pd.DataFrame):
        self.df_filtered = dataframe

    @abstractmethod
    def execute(self):
        raise NotImplementedError



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
