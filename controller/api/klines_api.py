import time
from binance.client import Client
import pandas as pd
from .utils import KlineUtils, KlineTimes
import pathlib


class KlineAPI:
    def __init__(self, symbol, interval, api="coin_margined"):
        self.client = Client()
        self.symbol = symbol
        self.interval = interval
        self.utils = KlineTimes(self.symbol, self.interval)
        self.klines = None
        self.api = api.lower()
        self.api_list = ["coin_margined", "mark_price", "spot"]
        if self.api not in self.api_list:
            raise ValueError(
                "Klines function should be either" "'coin_margined', 'mark_price' or 'spot'"
            )

    def get_exchange_symbol_info(self):
        if self.api == "mark_price":
            raise ValueError("Mark Price doesn't have a exchange simbol info")
        if self.api == "coin_margined":
            info = self.client.futures_coin_exchange_info()
        else:
            info = self.client.get_exchange_info()

        info_df = pd.DataFrame(info["symbols"])
        symbol_info = info_df.query(f"symbol == '{self.symbol}'")
        return symbol_info

    def get_ticker_info(self):
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

    def get_tick_size(self):
        df = self.get_ticker_info()
        tick_size = df.loc["PRICE_FILTER", "tickSize"]
        return tick_size

    def request_klines(
        self,
        startTime,
        endTime,
    ):
        if self.api == "coin_margined":
            api_get_klines = self.client.futures_coin_klines
        elif self.api == "mark_price":
            api_get_klines = self.client.futures_coin_mark_price_klines
        else:  # spot
            api_get_klines = self.client.get_klines

        request = api_get_klines(
            symbol=self.symbol,
            interval=self.interval,
            startTime=startTime,
            endTime=endTime,
            limit=1500,
        )
        return request

    def get_Klines(
        self,
        start_time=1502942400000,
    ):
        if (
            self.api == "coin_margined" or self.api == "mark_price"
        ) and start_time < 1597118400000:
            start_time = 1597118400000

        klines_list = []
        end_times = self.utils.get_end_times(start_time)

        START = time.time()

        for index in range(0, len(end_times) - 1):
            klines_list.extend(
                self.request_klines(
                    int(end_times[index]),
                    int(end_times[index + 1]),
                )
            )
            print("\nQty  : " + str(len(klines_list)))

        print(time.time() - START)
        self.klines = klines_list
        return self

    def update_data(self):
        data_path = pathlib.Path("model", "data")
        data_name = f"{self.symbol}_{self.interval}_{self.api}.parquet"
        dataframe_path = data_path.joinpath(data_name)
        data_frame = pd.read_parquet(dataframe_path)
        last_time = data_frame["open_time_ms"][-1]
        new_dataframe = self.get_Klines(last_time).to_OHLC_DataFrame()
        old_dataframe = data_frame.iloc[:-1, :]
        refresh_dataframe = pd.concat([old_dataframe, new_dataframe])
        self.klines = refresh_dataframe.copy()
        return self.klines

    def to_DataFrame(self):
        klines_df = KlineUtils(self.klines).klines_df()
        self.klines = klines_df.copy()
        return self.klines

    def to_OHLC_DataFrame(self):
        klines_df = KlineUtils(self.klines).klines_df()
        ohlc_columns = klines_df.columns[0:4].to_list()
        open_time_column = klines_df.columns[-1]
        klines_df = klines_df[ohlc_columns + [open_time_column]]

        self.klines = klines_df.copy()
        return self.klines
