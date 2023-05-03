import time
from binance.client import Client
import pandas as pd
import numpy as np
from binance.helpers import interval_to_milliseconds
from .utils import Klines
from math import ceil


class CoinMargined:
    def __init__(self, symbol, interval):
        self.client = Client()
        self.symbol = symbol
        self.interval = interval

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
        df = self.get_ticker_info(self.symbol)
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

    def calculate_max_multiplier(self):
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

    def get_All_Klines(
        self,
        start_time=1597118400000,
    ):
        START = time.time()
        time_delta = (time.time() * 1000 - start_time)
        time_delta_ratio = time_delta / interval_to_milliseconds(self.interval)
        request_qty = time_delta_ratio / self.calculate_max_multiplier()

        #get list for all end_times
        end_times = np.empty((ceil(request_qty)))
        end_times = np.arange(ceil(request_qty)) * (time_delta / request_qty) + start_time
        end_times = np.append(end_times,(time.time() * 1000))

        klines_list = []

        for index in range(0,len(end_times) - 1):
            klines_list.extend(
                self.futures_Kline(
                    int(end_times[index]),
                    int(end_times[index+1]),
                )
            )
            print("\nQty  : " + str(len(klines_list)))

        print(time.time() - START)
        return klines_list

    def get_Historical_Klines(
        self,
        first_Candle_Time=1597118400000,
    ):
        START = time.time()
        klines_list = []
        timeLoop_list = []
        index = 0
        initial_Time = first_Candle_Time
        interval_ms = interval_to_milliseconds(self.interval)
        max_Interval = interval_ms * 1500
        initial_Time = initial_Time - max_Interval
        while True:
            index += 1
            initial_Time += max_Interval
            timeLoop_list.append(initial_Time)
            if timeLoop_list[-1] + max_Interval < int(time.time() * 1000):
                request_Time_Start = time.time()
                klines_Loop = self.futures_Kline(
                    timeLoop_list[index - 1],
                    timeLoop_list[index - 1] + max_Interval,
                )
                klines_list.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQty  : " + str(len(klines_list)))
                request_Time_End = time.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    time.sleep(1.33 - request_Duration)
            else:
                print("\nLoop Ended\n")

                END = time.time()
                print("\nExecution time: " + str(END - START))
                break
        return klines_list

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

    def get_markPrice_All_Klines(
        self,
        first_Candle_Time=1597118400000,
    ):
        START = time.time()
        kline_List = []
        timeLoop = []
        index = 0
        initial_Time = first_Candle_Time
        max_multiplier = self.calculate_max_multiplier()
        max_Interval = interval_to_milliseconds(self.interval) * max_multiplier
        initial_Time = initial_Time - max_Interval

        while True:
            initial_Time += max_Interval
            index += 1
            timeLoop.append(initial_Time)
            if timeLoop[-1] + max_Interval < int(time.time() * 1000):
                request_Time_Start = time.time()
                klines_Loop = self.markPrice_futures_Kline(
                    timeLoop[index - 1],
                    timeLoop[index - 1] + max_Interval,
                )
                kline_List.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQty  : " + str(len(kline_List)))
                request_Time_End = time.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    time.sleep(1.33 - request_Duration)
            else:
                print("Else Reached!")
                lastCall = self.markPrice_futures_Kline(timeLoop[-1] + 1, int(time.time() * 1000))
                kline_List.extend(lastCall)
                print("\nQty  : " + str(len(kline_List)))
                print("\nLoop Ended\n")

                END = time.time()
                print("\nExecution time: " + str(END - START))
                break
        return kline_List

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
