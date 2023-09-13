import time
from typing import Literal
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
        symbol:str,
        interval:str,
        exchange:ccxt.Exchange = ccxt.bitstamp(),
        first_candle_time:int = 1325296800000,
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

    def _fetch_klines(self, since, limit: int=None) -> list:
        return self.exchange.fetch_ohlcv(
            symbol=self.symbol,
            timeframe=self.interval,
            since=since,
            limit=limit,
        )

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
            klines = self._fetch_klines(
                since=int(end_times[index]),
                limit=self.max_multiplier,
            )

            load_percentage = ((index / (len(end_times) - 1)) * 100)
            print(
                f"Finding first candle time [{load_percentage:.2f}%]"
            )

            if len(klines) > 0:
                print("Finding first candle time [100%]")
                first_unix_time = klines[0][0]
                print(f"\nFirst candle time found: {first_unix_time}\n")
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
        not_supported_types = (
            type(ccxt.bittrex()),
            type(ccxt.gemini())
        )

        if isinstance(self.exchange, not_supported_types):
            raise ValueError(f"{self.exchange} is not supported")

        klines = []
        klines_list = []

        first_call = self._fetch_klines(self.first_candle_time)

        if first_call:
            first_unix_time = first_call[0][0]
        else:
            first_unix_time = self.search_first_candle_time()

        START = time.perf_counter()


        temp_end_klines = None

        last_candle_interval = (
            time.time() * 1000 - interval_to_milliseconds(self.interval)
        )

        print("Starting requests \n")

        while True:
            time_value = klines[-1][0] + 1 if klines else first_unix_time
            klines = self._fetch_klines(time_value)
            klines_list.extend(klines)

            if klines == []:
                break
            if klines_list[-1][0] >= last_candle_interval:
                print("Qty  : " + str(len(klines_list)))
                break

            if temp_end_klines:
                if temp_end_klines == klines[-1][0]:
                    raise ValueError("End time not found")
            else:
                temp_end_klines = klines[-1][0]
            print("Qty  : " + str(len(klines_list)))

        print(f"\nElapsed time: {time.perf_counter() - START}\n")
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

    def aggregate_klines(
        self,
        exchanges: list[ccxt.Exchange] = None,
        symbols: list[str] = None,
        output_format: Literal["DataFrame", "Kline", "Both"] = "DataFrame",
        method: Literal["mean", "median", "hilo-mean", "hilo-median"] = "mean",
    ) -> pd.DataFrame | dict | tuple:
        """
        Aggregate the fetched K-line data into a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame
            Returns a pandas DataFrame containing K-line data.
        """
        if exchanges is None:
            exchanges = [ccxt.binance(), ccxt.bitstamp()]
        if symbols is None:
            symbols = ["BTC/USDT", "BTC/USD"]

        if method not in ["mean", "median", "hilo-mean", "hilo-median"]:
            raise ValueError("Invalid method argument")
        if output_format not in ["DataFrame", "Kline", "Both"]:
            raise ValueError("Invalid output format argument")

        aggregated_klines = {}

        for exchange, symbol in zip(exchanges, symbols):
            self.exchange = exchange
            self.symbol = symbol
            aggregated_klines[self.exchange.name] = (
                self.get_all_klines()
                .to_OHLCV()
                .data_frame
            )

        aggregated_df = (
            pd.concat(aggregated_klines.values(), axis=0)
            .groupby('date')
        )

        if method == "mean":
            aggregated_df = aggregated_df.mean()
        elif method == "median":
            aggregated_df = aggregated_df.median()
        elif method == "hilo-mean":
            aggregated_df = aggregated_df.agg(
                {
                    'open': 'mean',
                    'high': 'max',
                    'low': 'min',
                    'close': 'mean',
                }
            )
        elif method == "hilo-median":
            aggregated_df = aggregated_df.agg(
                {
                    'open': 'median',
                    'high': 'max',
                    'low': 'min',
                    'close': 'median',
                }
            )

        if output_format == "DataFrame":
            return aggregated_df
        if output_format == "Kline":
            return aggregated_klines
        if output_format == "Both":
            return aggregated_df, aggregated_klines

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
