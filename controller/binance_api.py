import time as t
from binance.client import Client

# spot Klines
class spot_API:
    def __init__(self, api_key, secret_key):
        self.client = Client(api_key, secret_key)

    def get_Spot_Kline(
        self, startTime, endTime, interval='2h', symbol="BTCUSDT"
    ):
        request = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=1000,
        )
        return request

    def get_All_Klines(self, interval, interval_ms, first_Candle_Time=1502942400000):
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
                )
                klines_list.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQtd  : " + str(len(klines_list)))
                request_Time_End = t.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
            else:
                print("Else Reached!")
                lastCall = self.get_Spot_Kline(
                    timeLoop_list[-1] + 1, int(t.time() * 1000), interval
                )
                klines_list.extend(lastCall)
                print("\nQtd  : " + str(len(klines_list)))
                print("\nLoop Finalizado\n")

                END = t.time()
                print("\nExecution time: " + str(END - START))
                break
        return klines_list

    def get_Historical_Klines(
        self, interval, interval_ms, first_Candle_Time=1502942400000
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
                )
                klines_list.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQtd  : " + str(len(klines_list)))
                request_Time_End = t.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
            else:
                print("\nLoop Finalizado\n")

                END = t.time()
                print("\nExecution time: " + str(END - START))
                print(timeLoop_list[index - 1])
                print(timeLoop_list[-1])
                break
        return klines_list


class source:
    def get_Source_List(self,klines):
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
    def __init__(self, api_key, secret_key):
        self.client = Client(api_key, secret_key)

    def futures_Kline(self, startTime, endTime, interval='2h'):
        request = self.client.futures_coin_klines(
            symbol="BTCUSD_PERP",
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=1500,
        )
        return request

    def get_All_Klines(self, interval, interval_ms, first_Candle_Time=1597118400000):
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
                )
                klines_list.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQtd  : " + str(len(klines_list)))
                request_Time_End = t.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
            else:
                print("Else Reached!")
                lastCall = self.futures_Kline(timeLoop_list[-1] + 1, "", interval)
                klines_list.extend(lastCall)
                print("\nQtd  : " + str(len(klines_list)))
                print("\nLoop Finalizado\n")

                END = t.time()
                print("\nExecution time: " + str(END - START))
                break
        return klines_list

    def get_Historical_Klines(
        self, interval, interval_ms, first_Candle_Time=1597118400000
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
                )
                klines_list.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQtd  : " + str(len(klines_list)))
                request_Time_End = t.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
            else:
                print("\nLoop Finalizado\n")

                END = t.time()
                print("\nExecution time: " + str(END - START))
                break
        return klines_list

    def markPrice_futures_Kline(
        self, startTime, endTime, interval='2h'
    ):
        request = self.client.futures_coin_mark_price_klines(
            symbol="BTCUSD_PERP",
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=1500,
        )
        return request

    def get_markPrice_All_Klines(
        self, interval, interval_ms, first_Candle_Time=1597118400000
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
                    timeLoop[index - 1], timeLoop[index - 1] + max_Interval, interval
                )
                kline_List.extend(klines_Loop)
                print("\nLoop : " + str(index))
                print("\nQtd  : " + str(len(kline_List)))
                request_Time_End = t.time()
                request_Duration = request_Time_End - request_Time_Start
                if request_Duration < 1.33:
                    t.sleep(1.33 - request_Duration)
            else:
                print("Else Reached!")
                lastCall = self.markPrice_futures_Kline(timeLoop[-1] + 1, "", interval)
                kline_List.extend(lastCall)
                print("\nQtd  : " + str(len(kline_List)))
                print("\nLoop Finalizado\n")

                END = t.time()
                print("\nExecution time: " + str(END - START))
                break
        return kline_List


# Classe ainda não testada
class coin_Trade:
    def __init__(self, api_key, secret_key):
        self.client = Client(api_key, secret_key)

    def get_open_orders(self):
        return self.client.futures_coin_get_open_orders()

    def get_all_orders(self):
        return self.client.futures_coin_get_all_orders(symbol="BTCUSD_PERP")

    def stop_Buy(self, price, quantity):
        return self.client.futures_coin_create_order(
            symbol="BTCUSD_PERP",
            side="BUY",
            type="STOP_MARKET",
            stopPrice=price,
            quantity=quantity,
        )

    def stop_Market(self, stopPrice):
        # Stop Market
        return self.client.futures_coin_create_order(
            symbol="BTCUSD_PERP",
            side="SELL",
            type="STOP_MARKET",
            stopPrice=stopPrice,
            closePosition="true",
            priceProtect="TRUE",
        )

    def buy_Market(self, quantity: int):
        # buy at Market Price
        return self.client.futures_coin_create_order(
            symbol="BTCUSD_PERP",
            side="BUY",
            type="MARKET",
            quantity=quantity,
        )

    def take_Profit(self, take_Profit):
        # Take Profit
        return self.client.futures_create_order(
            symbol="BTCUSD_PERP",
            side="SELL",
            type="LIMIT",
            timeInForce="GTC",
            stopPrice=take_Profit,
            closePosition="true",
            priceProtect="TRUE",
        )

    def buy_Position(self, contracts, entryPrice, profitPrice, stopPrice):
        entry = {
            "symbol": "BTCUSD_PERP",
            "side": "BUY",
            "type": "STOP_MARKET",
            "stopPrice": str(entryPrice),
            "quantity": str(contracts),
        }

        take_Profit = {
            "symbol": "BTCUSD_PERP",
            "side": "SELL",
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": str(profitPrice),
            "closePosition": "True",
        }

        stop_Loss = {
            "symbol": "BTCUSD_PERP",
            "side": "SELL",
            "type": "STOP_MARKET",
            "stopPrice": str(stopPrice),
            "closePosition": "True",
        }

        position_Order = []
        position_Order.append(entry)
        position_Order.append(take_Profit)
        position_Order.append(stop_Loss)
        return self.client.futures_coin_place_batch_order(batchOrders=position_Order)

    def sell_Position(self, contracts, entryPrice, profitPrice, stopPrice):
        entry = {
            "symbol": "BTCUSD_PERP",
            "side": "SELL",
            "type": "STOP_MARKET",
            "stopPrice": str(entryPrice),
            "quantity": str(contracts),
        }

        take_Profit = {
            "symbol": "BTCUSD_PERP",
            "side": "BUY",
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": str(profitPrice),
            "closePosition": "True",
        }

        stop_Loss = {
            "symbol": "BTCUSD_PERP",
            "side": "BUY",
            "type": "STOP_MARKET",
            "stopPrice": str(stopPrice),
            "closePosition": "True",
        }

        position_Order = []
        position_Order.append(entry)
        position_Order.append(take_Profit)
        position_Order.append(stop_Loss)
        return self.client.futures_coin_place_batch_order(batchOrders=position_Order)

    def take_Profit_Market(self, price):
        self.client.futures_coin_create_order(
            symbol="BTCUSD_PERP",
            side="SELL",
            type="TAKE_PROFIT_MARKET",
            stopPrice=price,
            closePosition="true",
            priceProtect="TRUE",
        )

    def limit(self, price):
        return self.client.futures_coin_create_order(
            symbol="BTCUSD_PERP",
            side="SELL",
            type="LIMIT",
            stopPrice=price,
            closePosition="true",
            priceProtect="TRUE",
        )

    def stop_Limit(self, price, stopPrice):
        return self.client.futures_coin_create_order(
            symbol="BTCUSD_PERP",
            side="SELL",
            type="STOP",
            stopPrice=stopPrice,
            price=price,
            closePosition="true",
            priceProtect="TRUE",
        )

    def cancel_All_Orders(self, symbol="BTCUSD_PERP"):
        return self.client.futures_coin_cancel_all_open_orders(symbol=symbol)

    def leverage(self, leverage, symbol="BTCUSD_PERP"):
        leverage_Request = self.client.futures_coin_change_leverage(
            symbol=symbol,
            leverage=leverage,
        )
        print("Leverage changed to: " + str(leverage))
        return leverage_Request

    def max_Leverage(self, trade_Risk, symbol="BTCUSD_PERP"):
        max_Leverage = round(1 / trade_Risk * 0.65)
        max_Leverage_Request = self.client.futures_coin_change_leverage(
            symbol=symbol,
            leverage=max_Leverage,
        )
        print("Leverage changed to: " + str(max_Leverage))
        return max_Leverage_Request

    def position_Info(self):
        return self.client.futures_coin_position_information()
