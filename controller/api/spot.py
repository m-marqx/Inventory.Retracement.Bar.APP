import time as t
from binance.client import Client
import pandas as pd
from binance.helpers import interval_to_milliseconds

class SpotAPI:
    def __init__(self):
        self.client = Client()

    def get_ticker_info(self,Symbol):
        info = self.client.get_exchange_info()
        info_df = pd.DataFrame(info['symbols'])
        filtered_info = [x['filters'] for x in info_df[info_df['symbol'] == Symbol].to_dict(orient='records')][0]
        df_filtered = pd.DataFrame.from_records(filtered_info)
        df_filtered.set_index('filterType',inplace=True)
        df_filtered = df_filtered.astype('float64')
        return df_filtered

    def get_Spot_Kline(
        self,
        startTime,
        endTime,
        interval="2h",
        symbol="BTCUSDT",
    ):
        request = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=1000,
        )
        return request

    def get_All_Klines(
        self,
        interval,
        first_Candle_Time=1502942400000,
        symbol="BTCUSDT",
        max_limit=1000,
        request_limit=False,
    ):
        START = t.time()
        klines_list = []
        timeLoop_list = []
        index = 0
        initial_Time = first_Candle_Time
        interval_ms = interval_to_milliseconds(interval)
        max_Interval = interval_ms * max_limit
        initial_Time = initial_Time - max_Interval
        while True:
            index += 1
            initial_Time += max_Interval
            timeLoop_list.append(initial_Time)
            if timeLoop_list[-1] + max_Interval < int(t.time() * max_limit):
                request_Time_Start = t.time()
                klines_Loop = self.get_Spot_Kline(
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
                lastCall = self.get_Spot_Kline(
                    timeLoop_list[-1] + 1,
                    int(t.time() * 1000),
                    interval,
                    symbol
                )
                klines_list.extend(lastCall)
                print("\nQty  : " + str(len(klines_list)))
                print("\nLoop Ended\n")

                END = t.time()
                print("\nExecution time: " + str(END - START))
                break
        return klines_list

    def get_Historical_Klines(
        self,
        interval,
        first_Candle_Time=1502942400000,
        symbol="BTCUSDT",
    ):
        START = t.time()
        klines_list = []
        timeLoop_list = []
        index = 0
        initial_Time = first_Candle_Time
        interval_ms = interval_to_milliseconds(interval)
        max_Interval = interval_ms * 1000
        initial_Time = initial_Time - max_Interval
        while True:
            index += 1
            initial_Time += max_Interval
            timeLoop_list.append(initial_Time)
            if timeLoop_list[-1] + max_Interval < int(t.time() * 1000):
                request_Time_Start = t.time()
                klines_Loop = self.get_Spot_Kline(
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
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
            else:
                print("\nLoop Ended\n")

                END = t.time()
                print("\nExecution time: " + str(END - START))
                print(timeLoop_list[index - 1])
                print(timeLoop_list[-1])
                break
        return klines_list