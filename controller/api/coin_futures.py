import time
from binance.client import Client
import pandas as pd
from .utils import Klines, KlineAnalyzer
import pathlib


class CoinMargined:
    def __init__(self, symbol, interval):
        self.client = Client()
        self.symbol = symbol
        self.interval = interval
        self.utils = KlineAnalyzer(self.symbol, self.interval)

    def get_ticker_info(self):
        info = self.client.futures_coin_exchange_info()
        info_df = pd.DataFrame(info["symbols"])
        filtered_info = (
            [x['filters'] for x in info_df[info_df['symbol'] == self.symbol]
            .to_dict(orient='records')][0]
        )
        df_filtered = pd.DataFrame.from_records(filtered_info)
        df_filtered.set_index("filterType", inplace=True)
        df_filtered = df_filtered.astype("float64")
        return df_filtered

    def get_tick_size(self):
        df = self.get_ticker_info()
        tick_size = df.loc["PRICE_FILTER", "tickSize"]
        return tick_size

    def futures_Kline(
        self,
        startTime,
        endTime,
    ):
        request = self.client.futures_coin_klines(
            symbol=self.symbol,
            interval=self.interval,
            startTime=startTime,
            endTime=endTime,
            limit=1500,
        )
        return request

    def markPrice_futures_Kline(
        self,
        startTime,
        endTime,
    ):
        request = self.client.futures_coin_mark_price_klines(
            symbol=self.symbol,
            interval=self.interval,
            startTime=startTime,
            endTime=endTime,
            limit=1500,
        )
        return request

    def spot_Kline(
        self,
        startTime,
        endTime,
    ):
        request = self.client.get_klines(
            symbol=self.symbol,
            interval=self.interval,
            startTime=startTime,
            endTime=endTime,
            limit=1000,
        )
        return request

    def get_All_Klines(
        self,
        start_time=1502942400000,
        klines_function="futures",
    ):

        klines_function = klines_function.lower()
        if (
            (klines_function == "futures" or klines_function == "mark price")
            and start_time < 1597118400000
        ):

            start_time = 1597118400000

        if klines_function == "futures":
            klines_function = self.futures_Kline
        elif klines_function == "mark price":
            klines_function = self.markPrice_futures_Kline
        elif klines_function == "spot":
            klines_function = self.spot_Kline
        else:
            raise TypeError(
                "Klines function should be either"
                "'futures', 'mark price' or 'spot'"
            )
        klines_list = []
        end_times = self.utils.get_end_times(start_time)

        START = time.time()

        for index in range(0,len(end_times) - 1):
            klines_list.extend(
                klines_function(
                    int(end_times[index]),
                    int(end_times[index+1]),
                )
            )
            print("\nQty  : " + str(len(klines_list)))

        print(time.time() - START)
        return klines_list

    def update_data(self):
        data_path = pathlib.Path("model","data")
        data_name = f"{self.symbol}_{self.interval}.parquet"
        dataframe_path = data_path.joinpath(data_name)
        data_frame = pd.read_parquet(dataframe_path)
        last_time = data_frame["open_time_ms"][-1]
        new_data = self.get_All_Klines(last_time)
        new_dataframe = Klines(new_data).klines_df().reindex(columns=data_frame.columns)
        old_dataframe = data_frame.iloc[:-1,:]
        refresh_dataframe = pd.concat([old_dataframe,new_dataframe])
        return refresh_dataframe

    def get_all_futures_klines_df(self):
        klines_list = self.get_All_Klines()
        klines_df = Klines(klines_list).klines_df()
        ohlc_columns = klines_df.columns[0:4].to_list()
        open_time_column = klines_df.columns[-1]
        return klines_df[ohlc_columns + [open_time_column]]

    def get_all_futures_klines_df_complete(self):
        klines_list = self.get_All_Klines()
        klines_df = Klines(klines_list).klines_df()
        return klines_df
