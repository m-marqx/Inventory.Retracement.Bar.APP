import time
import itertools
import logging
from typing import Literal
import numpy as np
import pandas as pd
import ccxt
from model.utils import interval_to_milliseconds
from .utils import KlineTimes

logging.basicConfig(
    format='%(levelname)s %(asctime)s: %(message)s',
    datefmt='%H:%M:%S',
    force=True,
    level=logging.INFO,
)

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
        The CCXT exchange object
        (default: ccxt.bitstamp()).
    first_candle_time : int
        The Unix timestamp of the first candle
        (default: 1325296800000).
    verbose : bool
        If True, print verbose logging messages during data retrieval
        (default: False).

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
    max_multiplier : int
        The maximum multiplier calculated based on the time interval.

    Methods:
    --------
    search_first_candle_time() -> int or None:
        Search for the Unix timestamp of the first candle in the
        historical K-line data.

    get_all_klines(ignore_unsupported_exchanges=False) -> CcxtAPI:
        Fetch all K-line data for the specified symbol and interval.

    to_OHLCV() -> pd.DataFrame:
        Convert the fetched K-line data into a pandas DataFrame in
        OHLCV format.

    aggregate_klines(
        exchanges=None,
        symbols=None,
        output_format='DataFrame',
        method='mean',
        filter_by_largest_qty=True
    ) -> pd.DataFrame or dict or tuple:
        Aggregate the fetched K-line data into a pandas DataFrame.

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
        verbose:bool = False,
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
        verbose : bool
            If True, print verbose logging messages during data retrieval
            (default: False).
        """
        self.symbol = symbol
        self.interval = interval
        self.first_candle_time = first_candle_time
        self.exchange = exchange
        self.verbose = verbose
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
            if self.verbose:
                load_percentage = (index / (len(end_times) - 1)) * 100
                logging.info(
                    "Finding first candle time [%.2f%%]",
                    load_percentage
                )

            if len(klines) > 0:
                first_unix_time = klines[0][0]
                if self.verbose:
                    logging.info("Finding first candle time [100%]")
                    logging.info(
                        "First candle time found: %s\n",
                        first_unix_time
                    )
                break

        return first_unix_time

    def get_all_klines(
        self,
        until: int | None = None,
        ignore_unsupported_exchanges: bool = False
    ):
        """
        Fetch all K-line data for the specified symbol and interval
        using a for loop.

        Parameters:
        -----------
        until : None
            The end time for fetching K-line data.
        ignore_unsupported_exchanges : bool, optional
            If True, ignore exchanges that do not support the specified
            symbol.
            (default: False).

        Returns:
        --------
        CcxtAPI
            Returns the CcxtAPI object with the fetched K-line data.
        """
        if ignore_unsupported_exchanges:
            not_supported_types = None
        else:
            not_supported_types = (
                type(ccxt.bittrex()),
                type(ccxt.gemini()),
                type(ccxt.huobi()),
                type(ccxt.deribit()),
                type(ccxt.hitbtc()),
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

        last_candle_interval = (
            (
                time.time() * 1000 - interval_to_milliseconds(self.interval)
                if until is None
                else until
            )
        )

        if self.verbose:
            logging.info("Starting requests")

        time_value = klines[-1][0] + 1 if klines else first_unix_time
        time_delta = first_call[-1][0] - first_call[0][0]
        step = time_delta + pd.Timedelta(self.interval).value / 1e+6
        end_times = np.arange(time_value, last_candle_interval, step)

        for current_start_time in end_times:
            klines = self._fetch_klines(current_start_time)
            if not klines:
                break

            klines_list.extend(klines)

            if klines_list[-1][0] >= last_candle_interval:
                if self.verbose:
                    logging.info(
                        "Qty: %d - Total: 100%% complete",
                        len(klines_list)
                    )
                break

            if self.verbose:

                percentage = (
                    (np.where(end_times == current_start_time)[0][0] + 1)
                    / end_times.shape[0]
                ) * 100
                logging.info(
                    "Qty: %d - Total: %.2f%% complete",
                    len(klines_list), percentage
                )


        if self.verbose:
            logging.info(
                "Requests elapsed time: %s\n",
                time.perf_counter() - START
            )
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

    def update_klines(
        self,
        klines: list | pd.DataFrame
    ) -> list | pd.DataFrame:
        """
        Update historical Klines data with new data.

        This method is used to update existing historical Klines data
        with fresh data obtained from the exchange. The function
        identifies the last timestamp in the provided Klines data and
        retrieves new data starting from that point.

        Parameters:
        -----------
        klines : list or pd.DataFrame
            The historical Klines data to be updated. This can be
            either a list of lists or a pandas DataFrame.

        Returns:
        --------
        list or pd.DataFrame
            The updated Klines data with the new data appended.
        """
        last_time = (
            klines[-3][0] if isinstance(klines, list)
            else int(klines.index[-3].value / 1e+6)
        )

        capi = CcxtAPI(self.symbol, self.interval, self.exchange, last_time)
        new_data = capi.get_all_klines()

        if isinstance(klines, list):
            new_klines = new_data.klines_list
            new_data = [
                data for data in new_klines
                if data[0] not in (kline[0] for kline in klines)
            ]

            updated_klines = klines + new_data
            updated_klines.sort()

        else:
            new_df = new_data.to_OHLCV().data_frame
            updated_klines = klines.combine_first(new_df)
            updated_klines = updated_klines.sort_index()
            updated_klines = updated_klines[
                ~updated_klines.index.duplicated(keep='last')
            ]

        return updated_klines

    def aggregate_klines(
        self,
        exchanges: list[ccxt.Exchange] = None,
        symbols: list[str] = None,
        output_format: Literal["DataFrame", "Kline", "Both"] = "DataFrame",
        method: Literal["mean", "median", "hilo-mean", "hilo-median"] = "mean",
        filter_by: Literal[
            "first_symbol",
            "volume",
            "exchange_volume"
        ] = "first_symbol"
    ) -> pd.DataFrame | dict | tuple[pd.DataFrame, dict]:
        """
        Aggregate K-line data from multiple exchanges and symbols.

        This method fetches K-line data from a list of exchanges and
        symbols, aggregates it using the specified method, and returns
        the aggregated data in the chosen format. You can select the
        output to be a pandas DataFrame, a dictionary of DataFrames,
        or both.

        Parameters:
        -----------
        exchanges : list[ccxt.Exchange], optional
            A list of CCXT exchange objects to use for data aggregation.
            If None, default exchanges (Binance and Bitstamp) will be
            used.
            (default: None)
        symbols : list[str], optional
            A list of trading symbols to aggregate data for. If None,
            default symbols ('BTC/USDT' and 'BTC/USD') will be used.
            (default: None)
        output_format : Literal["DataFrame", "Kline", "Both"], optional
            The format of the output. 'DataFrame' returns a pandas
            DataFrame, 'Kline' returns a dictionary of DataFrames for
            each exchange-symbol pair, and 'Both' returns both.
            (default: "DataFrame")
        method : Literal["mean", "median", "hilo-mean", "hilo-median"],
        optional
            The aggregation method to use for calculating aggregated
            K-line data.
            (default: "mean")
        filter_by :
        Literal["first_symbol", "volume", "exchange_volume"], optional
            Filter the results by choosing the first symbol with data,
            selecting the symbol with the highest trading volume, or
            selecting the symbol with the highest total exchange volume.
            (default: "first_symbol")

        Returns:
        --------
        pd.DataFrame or dict or tuple
            Depending on the `output_format` parameter, returns
            aggregated K-line data in:

            - DataFrame if `output_format` is 'DataFrame'
            - dict if `output_format` is 'Kline'
            - tuple[pd.DataFrame, dict] if `output_format` is 'Both'
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

        exchange_symbol_combinations = list(
            itertools.product(exchanges, symbols)
        )

        aggregated_klines = {}
        printed_symbols = set()
        index = 0

        filter_by_largest_qty = bool(filter_by == "first_symbol")

        if not filter_by_largest_qty:
            klines_qty = {}

        for exchange, symbol in exchange_symbol_combinations:
            markets_info = exchange.load_markets()
            has_symbol = any(symbol in markets_info for symbol in symbols)
            if has_symbol:
                if filter_by_largest_qty:
                    if symbol in markets_info and exchange not in printed_symbols:
                        if self.verbose:
                            index += 1
                            load_percentage = index / len(exchanges) * 100
                            logging.info(
                                "requesting klines [%.2f%%]",
                                load_percentage
                            )
                            logging.info("request: %s - %s", exchange, symbol)

                        self.exchange = exchange
                        self.symbol = symbol
                        printed_symbols.add(exchange)
                        aggregated_klines[self.exchange.name] = (
                            self.get_all_klines()
                            .to_OHLCV()
                            .data_frame
                        )
                else:
                    if symbol in markets_info:
                        if self.verbose:
                            index += 1
                            load_percentage = (
                                index
                                / len(exchange_symbol_combinations)
                                * 100
                            )

                            logging.info(
                                "requesting klines [%.2f%%]",
                                load_percentage
                            )
                            logging.info(
                                "request: %s - symbol: %s\n",
                                exchange,
                                symbol
                            )

                        printed_symbols.add(exchange)

                        self.exchange = exchange
                        self.symbol = symbol
                        market = f"{self.exchange.name} - {self.symbol}"
                        aggregated_klines[market] = (
                            self.get_all_klines()
                            .to_OHLCV()
                            .data_frame
                        )

                        klines_qty[market] = (
                            self.data_frame
                            .shape
                        )
            else:
                raise ValueError(
                    f"{exchange} doesn't have any of the specified symbols"
                )
        if not printed_symbols:
            raise ValueError(
                "None of the exchanges support any of the specified symbols"
            )
        if self.verbose:
            logging.info("requesting klines [100]%")
            logging.info("all klines successfully retrieved")

        aggregated_df = (
            pd.concat(aggregated_klines)
        )

        if not filter_by_largest_qty:
            klines_qty_df = pd.DataFrame.from_dict(
                klines_qty,
                orient='index',
                columns=['shape', 'columns']
            )

            klines_qty_df['exchange'] = (
                klines_qty_df.index
                .str.split(' - ').str[0]
            )
            max_shape_indices = (
                klines_qty_df
                .groupby('exchange')['shape']
                .idxmax()
            )

            aggregated_df = (
                aggregated_df
                .loc[max_shape_indices]
            )

        aggregated_df = aggregated_df.groupby('date')

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

        if filter_by == "volume":
            aggregated_df = self.__filter_by_volume(aggregated_klines)
        if filter_by == "exchange_volume":
            aggregated_df = self.__filter_by_exchange_volume(aggregated_klines)

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

    def __filter_by_volume(self, klines: dict) -> pd.DataFrame:
        first_exchange = max(klines, key=lambda x: klines[x].shape[0])
        data_frame = klines[first_exchange].copy()

        for _, kline_data in klines.items():
            temp_klines_dataframe = kline_data.reindex(data_frame.index)

            volume_comparison = (
                temp_klines_dataframe["volume"]
                > data_frame["volume"]
            )

            data_frame = data_frame.combine_first(
                temp_klines_dataframe.where(volume_comparison)
            )
        return data_frame

    def __filter_by_exchange_volume(self, klines: dict) -> pd.DataFrame:
        first_exchange = max(klines, key=lambda x: klines[x].shape[0])
        data_frame = klines[first_exchange].copy()

        exchange_volumes = {}

        for exchange, kline_data in klines.items():
            exchange_volumes[exchange] = kline_data['volume'].sum()

        exchange_with_highest_volume = max(
            exchange_volumes,
            key=exchange_volumes.get
        )

        data_frame.update(klines[exchange_with_highest_volume])
        return data_frame
