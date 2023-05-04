import numpy as np
import pandas as pd
from binance.helpers import interval_to_milliseconds

class Klines:
    def __init__(self, klines_list):
        self.klines_list = klines_list

    def klines_df(self) -> pd.DataFrame:
        timestamp = ["open_time", "close_time"]

        float_column = [
            "open",
            "high",
            "low",
            "close",
            "quote_asset_volume",
            "taker_buy_quote_asset_volume",
        ]

        int_column = ["volume", "number_of_trades", "taker_buy_base_asset_volume"]

        columns = (
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        )

        dataframe = pd.DataFrame(self.klines_list, columns=columns)

        dataframe["open_time_ms"] = dataframe["open_time"]
        dataframe[timestamp] = dataframe[timestamp].astype("datetime64[ms]")
        dataframe[float_column] = dataframe[float_column].astype(float)
        dataframe[int_column] = dataframe[int_column].astype(int)
        dataframe.set_index("open_time", inplace=True)
        return dataframe

class KlineAnalyzer:
    def __init__(self, symbol, interval):
        self.symbol = symbol
        self.interval = interval

    def calculate_max_multiplier(self):
        if self.interval != "1M":
            interval_hours = interval_to_milliseconds(self.interval) / 1000 / 60 / 60
            max_multiplier_limit = 1500
            max_days_limit = 200

            total_time_hours = interval_hours * np.arange(max_multiplier_limit, 0, -1)
            time_total_days = total_time_hours / 24

            max_multiplier = max_multiplier_limit - np.argmax(
                time_total_days <= max_days_limit
            )
        else:
            max_multiplier = 6

        return max_multiplier
