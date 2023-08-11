from math import ceil
import time
import numpy as np
import pandas as pd

from model.utils import interval_to_milliseconds

class KlineUtils:
    """
    A class for handling Kline data.

    Parameters
    ----------
    klines_list : list
        The list of Kline data.

    Attributes
    ----------
    klines_list : list
        The list of Kline data.

    Methods
    -------
    klines_df()
        Convert the Kline data to a DataFrame.

    """
    def __init__(self, klines_list):
        """
        Initialize the KlineUtils object.

        Parameters:
        -----------
        klines_list : list
            The list of Kline data.
        """
        self.klines_list = klines_list

    @property
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
    """
    Class for working with Kline times.

    Parameters
    ----------
    symbol : str
        The symbol of the asset.
    interval : str
        The interval of the Kline data.

    Attributes
    ----------
    symbol : str
        The symbol of the asset.
    interval : str
        The interval of the Kline data.

    Methods
    -------
    default_intervals()
        Returns the list of default intervals.
    calculate_max_multiplier(max_candle_limit: int = 1500)
        Calculate the maximum multiplier based on the interval.
    get_end_times(start_time=1597118400000, max_candle_limit=1500)
        Get the end times for retrieving Kline data.
    interval_max_divisor()
        Returns the maximum divisor of the interval.

    """
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

    @property
    def default_intervals(self):
        """
        Returns the list of default intervals.

        Returns
        -------
        list of str
            The list of default intervals.

        """
        return [
            "1s",
            "1m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ]

    def calculate_max_multiplier(
        self,
        max_candle_limit: int = 1500,
    ):
        """
        Calculate the maximum multiplier based on the interval.

        Returns:
        --------
        int
            The maximum multiplier.
        """
        if self.interval != "1M":

            interval_hours = (
                interval_to_milliseconds(self.interval)
                / 1000
                / 60
                / 60
            )

            max_multiplier_limit = max_candle_limit
            max_days_limit = 200

            total_time_hours = (
                interval_hours
                * np.arange(max_multiplier_limit, 0, -1)
            )

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
        max_candle_limit=1500,
    ):
        """
        Get the end times for retrieving Kline data.

        Parameters:
        -----------
        start_time : int, optional
            The start time for retrieving Kline data in milliseconds.
            (default: 1597118400000)

        Returns:
        --------
        numpy.ndarray
            The array of end times.
        """
        time_delta = time.time() * 1000 - start_time
        time_delta_ratio = time_delta / interval_to_milliseconds(self.interval)
        request_qty = (
            time_delta_ratio
            / self.calculate_max_multiplier(max_candle_limit)
        )

        end_times = (
            np.arange(ceil(request_qty))
            * (time_delta / request_qty)
            + start_time
        )
        end_times = np.append(end_times, time.time() * 1000)

        return end_times

    @property
    def get_max_interval(self):
        """
        Returns the maximum interval of the interval.

        Returns
        -------
        int
            The maximum interval of the interval.

        Raises
        ------
        ValueError
            If no divisible value is found or if a float value is entered.

        """
        if self.interval[-1] == "m":
            interval_range = self.default_intervals[1:5]
        elif self.interval[-1] == "h":
            interval_range = self.default_intervals[5:11]
        elif self.interval[-1] == "d":
            interval_range = self.default_intervals[11:13]
        elif self.interval[-1] == "w":
            interval_range = self.default_intervals[13:14]
        elif self.interval[-1] == "M":
            interval_range = self.default_intervals[14:15]
        else:
            interval_range = [0]

        int_interval_list = [x[:-1] for x in interval_range]
        int_interval_list = [int(x) for x in int_interval_list]
        int_interval = int(self.interval[:-1])

        max_divisor = None

        for value in reversed(int_interval_list):
            if int_interval % value == 0:
                max_divisor = value
                break

        if max_divisor is None:
            raise ValueError(
                "No divisible value found. Perhaps you entered a float value?"
            )
        max_interval = str(max_divisor) + self.interval[-1]
        return max_interval
