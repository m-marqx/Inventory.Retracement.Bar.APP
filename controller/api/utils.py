import numpy as np
import time
import pandas as pd
from math import ceil
from binance.helpers import interval_to_milliseconds


class KlineUtils:
    def __init__(self, klines_list):
        """
        Initialize the KlineUtils object.

        Parameters:
        -----------
        klines_list : list
            The list of Kline data.
        """
        self.klines_list = klines_list

    def klines_df(self) -> pd.DataFrame:
        """
        Convert the Kline data to a DataFrame.

        Returns:
        --------
        pd.DataFrame
            The Kline data as a DataFrame.
        """
        timestamp = ["open_time", "close_time"]

        float_column = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_quote_asset_volume",
            "taker_buy_base_asset_volume",
        ]

        int_column = ["number_of_trades"]

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
        dataframe[float_column] = dataframe[float_column].astype("float64")
        dataframe[int_column] = dataframe[int_column].astype("int64")
        dataframe.set_index("open_time", inplace=True)
        return dataframe


class KlineTimes:
    def __init__(self, symbol, interval):
        """
        Initialize the KlineTimes object

        Parameters:
        -----------
        symbol : str
            The symbol of the asset.
        interval : str
            The interval of the Kline data.
        """
        self.symbol = symbol
        self.interval = interval

    def calculate_max_multiplier(self):
        """
        Calculate the maximum multiplier based on the interval.

        Returns:
        --------
        int
            The maximum multiplier.
        """
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

    def get_end_times(
        self,
        start_time=1597118400000,
    ):
        """
        Get the end times for retrieving Kline data.

        Parameters:
        -----------
        start_time : int, optional
            The start time for retrieving Kline data in milliseconds. (default: 1597118400000)

        Returns:
        --------
        numpy.ndarray
            The array of end times.
        """
        time_delta = time.time() * 1000 - start_time
        time_delta_ratio = time_delta / interval_to_milliseconds(self.interval)
        request_qty = time_delta_ratio / self.calculate_max_multiplier()

        end_times = (
            np.arange(ceil(request_qty)) * (time_delta / request_qty) + start_time
        )
        end_times = np.append(end_times, time.time() * 1000)

        return end_times
