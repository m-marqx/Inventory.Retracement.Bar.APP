import time
from binance.client import Client
import pandas as pd
import numpy as np
from binance.helpers import interval_to_milliseconds
from .utils import Klines


class CoinMargined:
    def __init__(self):
        self.client = Client()

    def get_ticker_info(self, symbol):
        info = self.client.futures_coin_exchange_info()
        info_df = pd.DataFrame(info["symbols"])
        filtered_info = (
            [x['filters'] for x in info_df[info_df['symbol'] == symbol]
            .to_dict(orient='records')][0]
        )
        df_filtered = pd.DataFrame.from_records(filtered_info)
        df_filtered.set_index("filterType", inplace=True)
        df_filtered = df_filtered.astype("float64")
        return df_filtered

    def get_tick_size(self, symbol):
        df = self.get_ticker_info(symbol)
        tick_size = df.loc["PRICE_FILTER", "tickSize"]
        return tick_size

    def futures_Kline(
        self,
        startTime,
        endTime,
        interval="2h",
        symbol="BTCUSD_PERP",
    ):
        request = self.client.futures_coin_klines(
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=1500,
        )
        return request

    def calculate_max_multiplier(self, interval):
        if interval != "1M":
            interval_hours = interval_to_milliseconds(interval) / 1000 / 60 / 60
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
        interval="2h",
        first_Candle_Time=1597118400000,
        symbol="BTCUSD_PERP",
        request_limit=False,
    ):
        START = t.time()
        klines_list = []
        timeLoop_list = []
        index = 0
        initial_Time = first_Candle_Time

        if interval == "1M":
            intervalms = 2678000000
        else:
            intervalms = interval_to_milliseconds(interval)

        max_multiplier = self.calculate_max_multiplier(interval)
        max_Interval = intervalms * max_multiplier
        initial_Time = initial_Time - max_Interval

        while True:
            index += 1
            initial_Time += max_Interval
            timeLoop_list.append(initial_Time)
            if timeLoop_list[-1] + max_Interval < int(t.time() * 1000):
                request_Time_Start = t.time()
                klines_Loop = self.futures_Kline(
                    timeLoop_list[index - 1],
                    timeLoop_list[index - 1] + max_Interval,
                    interval,
                    symbol=symbol,
                )
                klines_list.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQty  : " + str(len(klines_list)))
                request_Time_End = t.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33 and request_limit:
                    t.sleep(1.33 - request_Duration)
            else:
                print("Else Reached!")
                lastCall = self.futures_Kline(timeLoop_list[-1] + 1, int(t.time() * 1000), interval)
                klines_list.extend(lastCall)
                print("\nQty  : " + str(len(klines_list)))
                print("\nLoop Ended\n")

                END = t.time()
                print("\nExecution time: " + str(END - START))
                break
        return klines_list

    def get_Historical_Klines(
        self,
        interval="2h",
        first_Candle_Time=1597118400000,
        symbol="BTCUSD_PERP",
    ):
        START = time.time()
        klines_list = []
        timeLoop_list = []
        index = 0
        initial_Time = first_Candle_Time
        interval_ms = interval_to_milliseconds(interval)
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
                    interval,
                    symbol=symbol,
                )
                klines_list.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQty  : " + str(len(klines_list)))
                request_Time_End = time.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
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
        interval="2h",
        symbol="BTCUSD_PERP",
    ):
        request = self.client.futures_coin_mark_price_klines(
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=1500,
        )
        return request

    def get_markPrice_All_Klines(
        self,
        interval,
        first_Candle_Time=1597118400000,
        symbol="BTCUSD_PERP",
    ):
        START = time.time()
        kline_List = []
        timeLoop = []
        index = 0
        initial_Time = first_Candle_Time
        max_multiplier = self.calculate_max_multiplier(interval)
        max_Interval = interval_to_milliseconds(interval) * max_multiplier
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
                    interval,
                    symbol=symbol,
                )
                kline_List.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQty  : " + str(len(kline_List)))
                request_Time_End = time.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
            else:
                print("Else Reached!")
                lastCall = self.markPrice_futures_Kline(timeLoop[-1] + 1, int(time.time() * 1000), interval)
                kline_List.extend(lastCall)
                print("\nQty  : " + str(len(kline_List)))
                print("\nLoop Ended\n")

                END = time.time()
                print("\nExecution time: " + str(END - START))
                break
        return kline_List

    def get_all_futures_klines_df(self, symbol, interval):
        klines_list = self.get_All_Klines(interval, symbol=symbol)
        klines_df = Klines(klines_list).klines_df()
        return klines_df
