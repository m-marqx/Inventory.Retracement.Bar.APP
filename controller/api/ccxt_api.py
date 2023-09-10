import time
import pandas as pd
import ccxt
from model.utils import interval_to_milliseconds
from .utils import KlineTimes

class CcxtAPI:
    """
    A class for interacting with the CCXT library to retrieve financial
    market data.

    Parameters:
    -----------
    symbol : str
        The trading symbol for the asset pair (e.g., 'BTC/USD').
    interval : str
        The time interval for K-line data
        (e.g., '1h' for 1-hour candles).
    exchange : ccxt.Exchange
        The CCXT exchange object (default: ccxt.bitstamp()).
    first_candle_time : int
        The Unix timestamp of the first candle
        (default: 1325296800000).

    Attributes:
    -----------
    symbol : str
        The trading symbol for the asset pair.
    interval : str
        The time interval for K-line data.
    first_candle_time : int
        The Unix timestamp of the first candle.
    data_frame : pd.DataFrame
        DataFrame to store the K-line data.
    exchange : ccxt.Exchange
        The CCXT exchange object.
    max_interval : str
        The maximum time interval supported by the asset pair.
    utils : KlineTimes
        An instance of the KlineTimes class for time-related
        calculations.

    Methods:
    --------
    get_all_klines():
        Fetch all K-line data for the specified symbol and interval.

    to_OHLCV() -> pd.DataFrame:
        Convert the fetched K-line data into a pandas DataFrame in
        OHLCV format.

    date_check() -> pd.DataFrame:
        Check for irregularities in the K-line data timestamps and
        return a DataFrame with discrepancies.

    """
    def __init__(
        self,
        symbol,
        interval,
        exchange=ccxt.bitstamp(),
        first_candle_time:int = 1325296800000
    ) -> None:
        """
        Initialize the CcxtAPI object.

        Parameters:
        -----------
        symbol : str
            The trading symbol for the asset pair.
        interval : str
            The time interval for K-line data.
        exchange : ccxt.Exchange
            The CCXT exchange object.
        first_candle_time : int
            The Unix timestamp of the first candle.
        """
        self.symbol = symbol
        self.interval = interval
        self.first_candle_time = first_candle_time
        self.exchange = exchange
        self.max_interval = KlineTimes(symbol, interval).get_max_interval
        self.utils = KlineTimes(self.symbol, self.max_interval)
        self.max_multiplier = int(self.utils.calculate_max_multiplier())
        self.data_frame = None
        self.klines_list = None

    def search_first_candle_time(self):
        """
        Search for the Unix timestamp of the first candle in the
        historical K-line data.

        This method iteratively fetches K-line data in reverse
        chronological order and stops when it finds the first candle.
        It can be used to determine the starting point for fetching
        historical data.

        Returns:
        --------
        int or None
            The Unix timestamp of the first candle found, or None
            if not found.
        """
        end_times = self.utils.get_end_times(
            self.first_candle_time,
            self.max_multiplier
        )

        for index in range(0, len(end_times) - 1):
            klines = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.interval,
                since=int(end_times[index]),
                limit=self.max_multiplier,
            )
            print(
                f"""Finding first candle time
                [{((index / (len(end_times) - 1)) * 100):.2f}%]"""
            )

            if len(klines) > 0:
                print("Finding first candle time [100%]")
                first_unix_time = klines[0][0]
                print(f"First candle time found: {first_unix_time}")
                break

        return first_unix_time

    def get_all_klines(self):
        """
        Fetch all K-line data for the specified symbol and interval.

        Returns:
        --------
        CcxtAPI
            Returns the CcxtAPI object with the fetched K-line data.
        """
        klines_list = []

        first_call = self.exchange.fetch_ohlcv(
            self.symbol,
            self.interval,
            since=self.first_candle_time,
            limit=self.max_multiplier
        )

        if first_call:
            first_unix_time = first_call[0][0]
        else:
            first_unix_time = self.search_first_candle_time()

        klines = []

        START = time.perf_counter()

        print("Starting loop")
        while True:
            time_value = klines[-1][0] + 1 if klines else first_unix_time

            klines = (
                self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.interval,
                    since=time_value,
                )
            )

            if 'temp_end_klines' in locals():
                if temp_end_klines == klines[-1][0]:
                    raise ValueError("End time not found")
            else:
                temp_end_klines = klines[-1][0]

            last_candle_interval = (
                time.time() * 1000 - interval_to_milliseconds("1d")
            )
            klines_list.extend(klines)

            print("\nQty  : " + str(len(klines_list)))

            if klines_list[-1][0] >= last_candle_interval:
                break


        print(f"\nElapsed time: {time.perf_counter() - START}")
        self.klines_list = klines_list
        return self

    def to_OHLCV(self) -> pd.DataFrame:
        """
        Convert the fetched K-line data into a pandas DataFrame in
        OHLCV format.

        Returns:
        --------
        pd.DataFrame
            Returns a pandas DataFrame containing OHLCV data.
        """
        ohlcv_columns = ["open", "high", "low", "close", "volume"]

        self.data_frame = pd.DataFrame(
            self.klines_list,
            columns=["date"] + ohlcv_columns
        )

        self.data_frame["date"] = self.data_frame["date"].astype(
            "datetime64[ms]"
        )

        self.data_frame = self.data_frame.set_index("date")
        return self

    def date_check(self) -> pd.DataFrame:
        """
        Check for irregularities in the K-line data timestamps and
        return a DataFrame with discrepancies.

        Returns:
        --------
        pd.DataFrame
            Returns a pandas DataFrame with discrepancies in
            timestamps.
        """
        ohlcv_columns = ["open", "high", "low", "close", "volume"]

        time_interval = pd.Timedelta(self.interval)

        date_check_df = self.data_frame.copy()
        date_check_df["actual_date"] = date_check_df.index
        date_check_df["previous_date"] = date_check_df["actual_date"].shift()

        date_check_df = date_check_df[
            ohlcv_columns
            + ["actual_date", "previous_date"]
        ]

        date_check_df["timedelta"] = (
            date_check_df["actual_date"]
            - date_check_df["previous_date"]
        )
        date_check_df = date_check_df.iloc[1:]

        date_check_df = date_check_df[
            date_check_df["timedelta"] != time_interval
        ]

        return date_check_df
