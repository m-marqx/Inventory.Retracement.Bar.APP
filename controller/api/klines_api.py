import time
import pathlib
import pandas as pd
from binance.client import Client
import model.utils.custom_pandas_methods
from controller.api.utils import KlineUtils, KlineTimes


class KlineAPI:
    """
    A class for retrieving and managing Kline data from an API.

    Parameters:
    -----------
    symbol : str
        The symbol for which to retrieve Kline data.
    interval : str
        The time interval for the Kline data
        (e.g., '1m', '5m', '1h', etc.).
    api : str, optional
        The API type to use for retrieving the data.
        Valid options are 'coin_margined', 'mark_price',
        or 'spot'.
        (default: 'coin_margined')

    Attributes:
    -----------
    symbol : str
        The symbol for which the Kline data is retrieved.
    interval : str
        The time interval of the Kline data.
    api : str
        The API type used for retrieving the data.
    client : Client
        The API client object.
    klines_list : list
        The list of Kline data.
    futures_options : list
        The list of API options for futures data.
    all_options : list
        The list of all API options.
    max_interval : int
        The maximum interval in milliseconds.
    utils : KlineTimes
        The KlineTimes object for utility functions.
    is_futures : bool
        Flag indicating if the API is for futures data.
    custom_interval : bool
        Flag indicating if the interval is custom.

    Methods:
    --------
    get_exchange_symbol_info()
        Get the exchange symbol information for the specified symbol.
    get_ticker_info()
        Get the ticker information for the specified symbol.
    get_tick_size()
        Get the tick size for the specified symbol.
    request_klines(start_time, end_time)
        Request Kline data for the specified time range.
    get_Klines(start_time)
        Get Kline data for the specified start time.
    update_data()
        Update the Kline data.
    update_klines(dataframe)
        Update the Kline data by appending new data to the existing
        DataFrame.
    to_DataFrame()
        Convert the Kline data to a DataFrame.
    to_OHLC_DataFrame()
        Convert the Kline data to an OHLC DataFrame.
    """
    def __init__(self, symbol, interval, api="coin_margined") -> None:
        """
        Initialize the KlineAPI object.

        Parameters:
        -----------
        symbol : str
            The symbol for which to retrieve Kline data.
        interval : str
            The time interval for the Kline data
            (e.g., '1m', '5m', '1h', etc.).
        api : str, optional
            The API type to use for retrieving the data.
            Valid options are 'coin_margined', 'mark_price',
            or 'spot'.
            (default: 'coin_margined')
        """
        self.symbol = symbol
        self.interval = interval
        self.api = api.lower()

        self.client = Client()
        self.klines_list = None
        self.futures_options = ["coin_margined" or "mark_price"]
        self.all_options = ["spot"] + self.futures_options

        self.max_interval = KlineTimes(self.symbol, interval).get_max_interval
        self.utils = KlineTimes(self.symbol, self.max_interval)

        self.is_futures = any(
            api_selected in self.futures_options
            for api_selected in [self.api]
        )

        if not any(
            api_selected in self.all_options
            for api_selected in [self.api]
        ):
            raise ValueError(
                "Klines function should be either "
                "'coin_margined', 'mark_price' or 'spot'"
            )

        self.custom_interval = interval not in self.utils.default_intervals

    def get_exchange_symbol_info(self) -> pd.DataFrame:
        """
        Get the exchange symbol information for the specified symbol.

        Returns:
        --------
        pd.DataFrame
            The exchange symbol information as a DataFrame.
        """
        if self.api == "mark_price":
            raise ValueError("Mark Price doesn't have an exchange symbol info")
        if self.api == "coin_margined":
            info = self.client.futures_coin_exchange_info()
        else:
            info = self.client.get_exchange_info()

        info_df = pd.DataFrame(info["symbols"])
        symbol_info = info_df.query(f"symbol == '{self.symbol}'")
        return symbol_info

    def get_ticker_info(self) -> pd.DataFrame:
        """
        Get the ticker information for the specified symbol.

        Returns:
        --------
        pd.DataFrame
            The ticker information as a DataFrame.
        """
        if self.api == "mark_price":
            raise ValueError("Mark Price doesn't have a ticker info")
        if self.api == "coin_margined":
            info = self.client.futures_coin_exchange_info()
        else:
            info = self.client.get_exchange_info()

        info_df = pd.DataFrame(info["symbols"])
        symbol_info = info_df.query(f"symbol == '{self.symbol}'")

        filters_info = symbol_info["filters"].explode().to_list()
        df_filtered = pd.DataFrame(filters_info)
        df_filtered.set_index("filterType", inplace=True)
        df_filtered = df_filtered.astype("float64")
        return df_filtered

    def get_tick_size(self) -> float:
        """
        Get the tick size for the specified symbol.

        Returns:
        --------
        float
            The tick size.
        """
        df = self.get_ticker_info()
        tick_size = df.loc["PRICE_FILTER", "tickSize"]
        return tick_size

    def request_klines(
        self,
        start_time,
        end_time,
    ) -> list:
        """
        Request Kline data for the specified time range.

        Parameters:
        -----------
        start_time : int
            The start time for the data range in milliseconds.
        end_time : int
            The end time for the data range in milliseconds.

        Returns:
        --------
        list
            The requested Kline data as a list.
        """
        if self.api == "coin_margined":
            api_get_klines = self.client.futures_coin_klines
        elif self.api == "mark_price":
            api_get_klines = self.client.futures_coin_mark_price_klines
        else:  # spot
            api_get_klines = self.client.get_klines

        request_limit = 1500

        if self.api == "spot":
            request_limit = 1000

        max_limit = self.utils.calculate_max_multiplier(request_limit)

        request = api_get_klines(
            symbol=self.symbol,
            interval=self.max_interval,
            startTime=start_time,
            endTime=end_time,
            limit=max_limit,
        )
        return request

    def get_Klines(
        self,
        start_time=1502942400000,
    ):
        """
        Get Kline data for the specified start time.

        Parameters:
        -----------
        start_time : int, optional
            The start time for retrieving Kline data in milliseconds.
            (default: 1502942400000)

        Returns:
        --------
        KlineAPI
            The KlineAPI object.
        """
        max_candle_limit = 1000

        if self.is_futures:
            start_time = max(start_time, 1597028400000)
            max_candle_limit = 1500

        end_times = self.utils.get_end_times(start_time, max_candle_limit)

        first_call = self.request_klines(
            int(end_times[0]),
            int(end_times[1]),
        )

        klines_list = first_call
        print("\nQty  : " + str(len(klines_list)))

        START = time.perf_counter()

        for index in range(1, len(end_times) - 1):
            klines_list.extend(
                self.request_klines(
                    int(end_times[index] + 1),
                    int(end_times[index + 1]),
                )
            )
            print("\nQty  : " + str(len(klines_list)))

        print(time.perf_counter() - START)
        self.klines_list = klines_list
        return self

    def update_data(
        self
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, None]:
        """
        Update the Kline data.

        Returns:
        --------
        pd.DataFrame
            The updated Kline data as a DataFrame.
        """
        data_path = pathlib.Path("model", "data")
        data_name = f"{self.symbol}_{self.max_interval}_{self.api}.parquet"
        dataframe_path = data_path.joinpath(data_name)
        data_frame = pd.read_parquet(dataframe_path)
        return self.update_klines(data_frame)

    def update_klines(
        self,
        dataframe
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, None]:
        """
        Update the Kline data by appending new data to the existing
        DataFrame.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The existing Kline data DataFrame.

        Returns:
        --------
        pd.DataFrame
            The updated Kline data DataFrame.
        """
        last_time = (
            (dataframe.index[-1] - pd.Timestamp("1970-01-01"))
            // pd.Timedelta('1ms')
        )

        new_dataframe = KlineUtils(
            self.get_Klines(last_time).klines_list
        ).klines_df[['open', 'high', 'low', 'close', 'open_time_ms']]

        old_dataframe = dataframe.iloc[:-1, :]
        updated_dataframe = pd.concat([old_dataframe, new_dataframe])

        if self.custom_interval:

            resampled_dataframe = (
                updated_dataframe
                .resample_ohlc(self.interval)
                .ohlc()
            )

            return updated_dataframe, resampled_dataframe

        return updated_dataframe, None

    def to_DataFrame(self) -> pd.DataFrame:
        """
        Convert the Kline data to a DataFrame.

        Returns:
        --------
        pd.DataFrame
            The Kline data as a DataFrame.
        """
        klines_df = KlineUtils(self.klines_list).klines_df

        if self.custom_interval:
            klines_df.resample_ohlc(self.interval).ohlc()
        return klines_df

    def to_OHLC_DataFrame(self) -> pd.DataFrame:
        """
        Convert the Kline data to an OHLC DataFrame.

        Returns:
        --------
        pd.DataFrame
            The Kline data as an OHLC DataFrame.
        """
        klines_df = KlineUtils(self.klines_list).klines_df
        ohlc_columns = klines_df.columns[0:4].to_list()
        open_time_column = klines_df.columns[-1]
        klines_df = klines_df[ohlc_columns + [open_time_column]]

        if self.custom_interval:
            klines_df = klines_df.resample_ohlc(self.interval).ohlc()
        return klines_df
