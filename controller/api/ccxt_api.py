import time
import pandas as pd
import ccxt
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
        self.data_frame = None
        self.klines_list = None

    def get_all_klines(self):
        """
        Fetch all K-line data for the specified symbol and interval.

        Returns:
        --------
        CcxtAPI
            Returns the CcxtAPI object with the fetched K-line data.
        """
        max_multiplier = int(self.utils.calculate_max_multiplier())
        first_call = self.exchange.fetch_ohlcv(
            self.symbol,
            self.interval,
            since=self.first_candle_time,
            limit=max_multiplier
        )

        if first_call:
            first_unix_time = first_call[0][0]
            end_times = self.utils.get_end_times(
                first_unix_time,
                max_multiplier
            )
        else:
            end_times = self.utils.get_end_times(1325296800000, max_multiplier)

        klines_list = []

        START = time.perf_counter()

        for index in range(0, len(end_times) - 1):
            klines_list.extend(
                self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.interval,
                    since=int(end_times[index]),
                    limit=max_multiplier,
                )
            )
            print("\nQty  : " + str(len(klines_list)))
        print(f"Elapsed time: {time.perf_counter() - START}")
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

