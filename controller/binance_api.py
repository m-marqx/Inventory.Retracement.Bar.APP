import time as t
from binance.client import Client
import pandas as pd
from controller import config

# spot Klines
class spot_API:
    def __init__(self):
        self.client = Client(config.api_key, config.secret_key)

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
        interval_ms,
        first_Candle_Time=1502942400000,
        symbol="BTCUSDT",
    ):
        START = t.time()
        klines_list = []
        timeLoop_list = []
        index = 0
        initial_Time = first_Candle_Time
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
                print("Else Reached!")
                lastCall = self.get_Spot_Kline(
                    timeLoop_list[-1] + 1, int(t.time() * 1000), interval, symbol=symbol
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
        interval_ms,
        first_Candle_Time=1502942400000,
        symbol="BTCUSDT",
    ):
        START = t.time()
        klines_list = []
        timeLoop_list = []
        index = 0
        initial_Time = first_Candle_Time
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


class source:
    def get_Source_List(self, klines):
        open_Time_List = []
        open_List = []
        high_List = []
        low_List = []
        close_List = []
        h2_List = []
        hlc3_List = []
        ohlc4_List = []
        volume_List = []
        close_Time_List = []

        for x in klines:
            open_Time_List.append(int(x[0]))
            open_List.append(float(x[1]))
            high_List.append(float(x[2]))
            low_List.append(float(x[3]))
            close_List.append(float(x[4]))
            volume_List.append(float(x[5]))
            close_Time_List.append(int(x[6]))

            h2_List.append(float((float(x[2]) + float(x[3]))) / 2)
            hlc3_List.append(float((float(x[2]) + float(x[3]) + float(x[4])) / 3))
            ohlc4_List.append(
                float((float(x[1]) + float(x[2]) + float(x[3]) + float(x[4])) / 4)
            )

        kline_Values = {
            "Open Time": open_Time_List,
            "Close Time": close_Time_List,
            "Open": open_List,
            "High": high_List,
            "Low": low_List,
            "Close": close_List,
            "Volume": volume_List,
            "H2": h2_List,
            "HLC3": hlc3_List,
            "OHLC4": ohlc4_List,
        }
        return kline_Values


# coin-M Klines
class futures_API:
    def __init__(self):
        self.client = Client(config.api_key, config.secret_key)

    def get_ticker_info(self,Symbol):
        info = self.client.futures_coin_exchange_info()
        info_df = pd.DataFrame(info['symbols'])
        filtered_info = [x['filters'] for x in info_df[info_df['symbol'] == Symbol].to_dict(orient='records')][0]
        df_filtered = pd.DataFrame.from_records(filtered_info)
        df_filtered.set_index('filterType')
        df_filtered.set_index('filterType',inplace=True)
        df_filtered['minPrice'] = df_filtered['minPrice'].astype('float64')
        df_filtered['maxPrice'] = df_filtered['maxPrice'].astype('float64')
        df_filtered['tickSize'] = df_filtered['tickSize'].astype('float64')
        df_filtered['stepSize'] = df_filtered['stepSize'].astype('float64')
        df_filtered['maxQty'] = df_filtered['maxQty'].astype('float64')
        df_filtered['minQty'] = df_filtered['minQty'].astype('float64')
        df_filtered['limit'] = df_filtered['limit'].astype('float64')
        df_filtered['multiplierDown'] = df_filtered['multiplierDown'].astype('float64')
        df_filtered['multiplierUp'] = df_filtered['multiplierUp'].astype('float64')
        df_filtered['multiplierDecimal'] = df_filtered['multiplierDecimal'].astype('float64')
        return df_filtered
    
    def get_tick_size(self,Symbol):
        
        df = self.get_ticker_info(Symbol)
        tick_size = df.loc['PRICE_FILTER', 'tickSize']
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
        interval,
        interval_ms,
        first_Candle_Time=1597118400000,
        symbol="BTCUSD_PERP",
    ):
        START = t.time()
        klines_list = []
        timeLoop_list = []
        index = 0
        initial_Time = first_Candle_Time
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
        interval,
        interval_ms,
        first_Candle_Time=1597118400000,
        symbol="BTCUSD_PERP",
    ):
        START = t.time()
        klines_list = []
        timeLoop_list = []
        index = 0
        initial_Time = first_Candle_Time
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
        interval_ms,
        first_Candle_Time=1597118400000,
        symbol="BTCUSD_PERP",
    ):
        START = t.time()
        kline_List = []
        timeLoop = []
        index = 0
        initial_Time = first_Candle_Time
        max_Interval = interval_ms * 1500
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

    def get_all_futures_klines_df(self,symbol, interval, intervalms):
        klines_list = self.get_All_Klines(interval, intervalms, symbol=symbol)
        timestamp = ['open_time','close_time']

        float_column = [
            'open','high','low','close',
            'quote_asset_volume',
            'taker_buy_quote_asset_volume'
        ]

        int_column = [
            'volume',
            'number_of_trades',
            'taker_buy_base_asset_volume'
        ]

        columns=(
            'open_time',
            'open','high','low','close','volume',
            'close_time',
            'quote_asset_volume',
            'number_of_trades',
            'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume',
            'ignore',
        )

        dataframe = pd.DataFrame(klines_list,columns=columns)

        dataframe[timestamp] = dataframe[timestamp].astype('datetime64[ms]')
        dataframe[float_column] = dataframe[float_column].astype(float)
        dataframe[int_column] = dataframe[int_column].astype(int)
        dataframe.set_index('open_time', inplace=True)
        return dataframe

    def get_df_to_csv(self,dataframe,name):
        str_name = f'{name}.csv'
        columns = [
            'open','high','low','close','volume',
            'close_time',
            'quote_asset_volume',
            'number_of_trades',
            'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume'
            ]
        dataframe.to_csv(
            f'../model/data/{str_name}', 
            index=True, 
            header=columns, 
            sep=';', 
            decimal='.', 
            encoding='utf-8')
        
        return print(str_name+' has been saved')