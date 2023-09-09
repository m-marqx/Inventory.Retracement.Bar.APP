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

