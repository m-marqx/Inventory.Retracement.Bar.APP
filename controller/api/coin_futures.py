import time as t
from binance.client import Client
import pandas as pd
from binance.helpers import interval_to_milliseconds

class coin_margined:
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
        intervalms = interval_to_milliseconds(interval)
        max_Interval = intervalms * 1500
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
                lastCall = self.futures_Kline(timeLoop_list[-1] + 1, "", interval)
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
        START = t.time()
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
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
            else:
                print("\nLoop Ended\n")

                END = t.time()
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
        START = t.time()
        kline_List = []
        timeLoop = []
        index = 0
        initial_Time = first_Candle_Time
        max_Interval = interval_to_milliseconds(interval_ms) * 1500
        initial_Time = initial_Time - max_Interval
        while True:
            initial_Time += max_Interval
            index += 1
            timeLoop.append(initial_Time)
            if timeLoop[-1] + max_Interval < int(t.time() * 1000):
                request_Time_Start = t.time()
                klines_Loop = self.markPrice_futures_Kline(
                    timeLoop[index - 1],
                    timeLoop[index - 1] + max_Interval,
                    interval,
                    symbol=symbol,
                )
                kline_List.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQty  : " + str(len(kline_List)))
                request_Time_End = t.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
            else:
                print("Else Reached!")
                lastCall = self.markPrice_futures_Kline(timeLoop[-1] + 1, "", interval)
                kline_List.extend(lastCall)
                print("\nQty  : " + str(len(kline_List)))
                print("\nLoop Ended\n")

                END = t.time()
                print("\nExecution time: " + str(END - START))
                break
        return kline_List

    def get_all_futures_klines_df(self, symbol, interval):
        klines_list = self.get_All_Klines(interval, symbol=symbol)
        timestamp = ["open_time", "close_time"]

        float_column = [
            "open",
            "high",
            "low",
            "close",
            "quote_asset_volume",
            "taker_buy_quote_asset_volume",
        ]

        int_column = ["volume", "number_of_trades", "taker_buy_base_asset_volume"]

        columns = (
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        )

        dataframe = pd.DataFrame(klines_list, columns=columns)

        dataframe[timestamp] = dataframe[timestamp].astype("datetime64[ms]")
        dataframe[float_column] = dataframe[float_column].astype(float)
        dataframe[int_column] = dataframe[int_column].astype(int)
        dataframe.set_index("open_time", inplace=True)
        return dataframe

    def get_df_to_csv(self, dataframe, name):
        str_name = f"{name}.csv"
        columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        dataframe.to_csv(
            f"model/data/{str_name}",
            index=True,
            header=columns,
            sep=";",
            decimal=".",
            encoding="utf-8",
        )

        return print(str_name + " has been saved")
