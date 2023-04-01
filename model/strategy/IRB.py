import pandas as pd
import numpy as np
from model.indicators.moving_average import moving_average

ma = moving_average()

class IRB_Strategy:
    def __init__(self, dataframe):
        self.columns = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close"
        }
        try:
            self.df_filtered = dataframe[self.columns.values()].copy()
        except KeyError:
            self.df_filtered = dataframe[self.columns.keys()].copy()
            self.df_filtered.rename(columns=self.columns, inplace=True)

        self.df_filtered["open"] = self.df_filtered["open"].astype(float)
        self.df_filtered["high"] = self.df_filtered["high"].astype(float)
        self.df_filtered["low"] = self.df_filtered["low"].astype(float)
        self.df_filtered["close"] = self.df_filtered["close"].astype(float)

        self.open_price = self.df_filtered["open"]
        self.high_price = self.df_filtered["high"]
        self.low_price = self.df_filtered["low"]
        self.close_price = self.df_filtered["close"]

        self.candle_amplitude = None
        self.tick_size = None
        self.ema = None
        self.candle_downtail = None
        self.candle_uppertail = None
        self.bullish_calculation = None
        self.bearish_calculation = None
        self.irb_condition = None
        self.irb_signal = None
        self.buy_condition = None
        self.entry_price = None
        self.target = None
        self.stop_loss = None
        self.signal_arr = None
        self.entry_price_arr = None
        self.take_profit_arr = None
        self.stop_loss_arr = None
        self.high_price_arr = None
        self.low_price_arr = None
        self.close_position_arr = None
        self.signal_condition = None
        self.open_position = None
        self.is_close_position = None
        self.is_take_profit = None
        self.is_stop_loss = None
        self.profit = None
        self.loss = None
        self.df_fixed_results = None
        self.is_close_position = None
        self.is_take_profit = None
        self.is_stop_loss = None
        self.df_check = pd.DataFrame()
        self.df_signals = pd.DataFrame()

    def set_ema(self, length=20):
        self.ema = ma.ema(self.close_price, length)
        self.df_filtered["ema"] = self.ema

        return self

    def set_tick_size(self, tick_size=0.1):
        self.tick_size = tick_size

        return self

    def get_irb_signals(self, payoff, lowestlow=1):
        self.df_filtered["uptrend"] = (
            self.close_price
            >= self.df_filtered["ema"]
        )
        self.candle_amplitude = self.high_price - self.low_price

        self.candle_downtail = (
            np.minimum(self.open_price, self.close_price)
            - self.low_price
        )

        self.candle_uppertail = self.high_price - np.maximum(
            self.open_price,
            self.close_price,
        )

        # Analyze the downtail and uptail of the candle
        # Assign a value to the IRB_Condition column based on the value of the wick
        self.bullish_calculation = (
            self.candle_uppertail
            / self.candle_amplitude
        )

        self.bearish_calculation = (
            self.candle_downtail
            / self.candle_amplitude
        )

        self.df_filtered["IRB_Condition"] = np.where(
            self.df_filtered["uptrend"],
            self.bullish_calculation,
            self.bearish_calculation,
        )

        self.irb_condition = self.df_filtered["IRB_Condition"] >= 0.45
        self.buy_condition = self.irb_condition & self.df_filtered["uptrend"]

        self.df_filtered["Signal"] = np.where(self.buy_condition, 1, np.nan)
        self.df_filtered["Signal"].astype("float32")

        self.entry_price = self.df_filtered["high"].shift(1) + self.tick_size
        self.target = self.df_filtered["high"].shift(1) + (
            self.candle_amplitude.shift(1) * payoff
        )

        # Stop Loss is the lowest low of the last X candles
        self.stop_loss = (
            self.df_filtered["low"].rolling(lowestlow).min().shift()
            - self.tick_size
        )

        # If the lowest low is NaN, fill it with the cumulative minimum
        self.stop_loss = self.stop_loss.fillna(self.df_filtered["low"].cummin())

        self.df_filtered["Entry_Price"] = np.where(
            self.buy_condition, self.entry_price, np.nan
        )

        self.df_filtered["Take_Profit"] = np.where(
            self.buy_condition, self.target, np.nan
        )

        self.df_filtered["Stop_Loss"] = np.where(
            self.buy_condition, self.stop_loss, np.nan
        )

        return self

    def calculate_irb_signals(self):
        self.signal_arr = self.df_filtered["Signal"].to_numpy()
        self.entry_price_arr = self.df_filtered["Entry_Price"].to_numpy()
        self.take_profit_arr = self.df_filtered["Take_Profit"].to_numpy()
        self.stop_loss_arr = self.df_filtered["Stop_Loss"].to_numpy()
        self.high_price_arr = self.df_filtered["high"].to_numpy()
        self.low_price_arr = self.df_filtered["low"].to_numpy()
        self.close_position_arr = np.zeros(len(self.df_filtered), dtype=bool)

        for index in range(1, len(self.df_filtered)):
            prev_index = index - 1
            self.signal_condition = self.signal_arr[prev_index] == 1
            self.open_position = ~self.close_position_arr[index]
            if self.signal_condition & self.open_position:
                self.signal_arr[index] = self.signal_arr[prev_index]
                self.entry_price_arr[index] = self.entry_price_arr[prev_index]
                self.take_profit_arr[index] = self.take_profit_arr[prev_index]
                self.stop_loss_arr[index] = self.stop_loss_arr[prev_index]
                
                self.profit = (
                    self.high_price_arr[index] 
                    > self.take_profit_arr[index]
                )
                
                self.loss = (
                    self.low_price_arr[index] 
                    < self.stop_loss_arr[index]
                )

                if self.profit ^ self.loss:
                    self.close_position_arr[index] = True
                    self.signal_arr[index] = -1

        self.df_signals = pd.DataFrame(
            {
                "Signal": self.signal_arr,
                "Entry_Price": self.entry_price_arr,
                "Take_Profit": self.take_profit_arr,
                "Stop_Loss": self.stop_loss_arr,
                "Close_Position": self.close_position_arr,
                "high": self.high_price_arr,
                "low": self.low_price_arr,
            }
        )

        return self

    def check_error(self):
        columns = ["Signal", "Close_Position"]
        self.df_check = self.df_signals[columns].copy()
        self.df_check["Signal_Shifted"] = self.df_check["Signal"].shift(1)

        is_null_signal = self.df_check["Signal"].isnull()
        is_null_signal_shift = self.df_check["Signal_Shifted"].isnull()

        has_signal_error = (
            is_null_signal 
            & (self.df_check["Signal_Shifted"] == 1)
        )

        has_close_error = (
            is_null_signal_shift 
            & self.df_check["Close_Position"]
        )

        has_error = has_signal_error | has_close_error

        if has_error.any():
            print("Error Found")

        self.df_check["Error"] = has_error

        return self

    def calculate_results(self, verify_error=True):
        columns = [
            "Signal",
            "Entry_Price",
            "Take_Profit",
            "Stop_Loss",
            "high",
            "low",
            "Close_Position",
        ]
        self.df_results = self.df_signals[columns].copy()
        self.is_close_position = self.df_results["Close_Position"]

        self.is_take_profit = (
            self.df_results["high"]
            > self.df_results["Take_Profit"]
        )

        self.is_stop_loss = (
            self.df_results["low"] 
            < self.df_results["Stop_Loss"]
        )

        self.profit = (
            self.df_results["Take_Profit"] 
            - self.df_results["Entry_Price"]
        )

        self.loss = (
            self.df_results["Stop_Loss"] 
            - self.df_results["Entry_Price"]
        )

        self.df_results["Result"] = 0
        self.df_results["Result"] = np.where(
            self.is_close_position & self.is_take_profit,
            self.profit,
            self.df_results["Result"],
        )

        self.df_results["Result"] = np.where(
            self.is_close_position & self.is_stop_loss,
            self.loss,
            self.df_results["Result"],
        )

        self.df_results["Cumulative_Result"] = (
            self.df_results["Result"]
            .cumsum()
        )

        if verify_error:
            self.check_error()

        return self

    def calculate_fixed_pl_results(self, profit, loss, verify_error=False):
        columns = [
            "Signal",
            "Entry_Price",
            "Take_Profit",
            "Stop_Loss",
            "high",
            "low",
            "Close_Position",
        ]
        self.df_fixed_results = self.df_signals[columns].copy()
        self.is_close_position = self.df_fixed_results["Close_Position"]
        self.is_take_profit = (
            self.df_fixed_results["high"]
            > self.df_fixed_results["Take_Profit"]
        )
        self.is_stop_loss = (
            self.df_fixed_results["low"]
            < self.df_fixed_results["Stop_Loss"]
        )

        self.df_fixed_results["Result"] = 0
        self.df_fixed_results["Result"] = np.where(
            self.is_close_position & self.is_take_profit,
            profit,
            self.df_fixed_results["Result"],
        )

        self.df_fixed_results["Result"] = np.where(
            self.is_close_position & self.is_stop_loss,
            -loss,
            self.df_fixed_results["Result"],
        )

        self.df_fixed_results["Cumulative_Result"] = (
            self
            .df_fixed_results["Result"]
            .cumsum()
        )

        if verify_error:
            self.check_error()

        return self

    def run_IRB_model(
        self,
        payoff,
        length,
        tick_size,
    ):
        strategy = (
            self.set_ema(length)
            .set_tick_size(tick_size)
            .get_irb_signals(payoff)
            .calculate_irb_signals()
            .calculate_results()
        )

        return strategy.df_results

    def run_IRB_model_fixed(
        self,
        payoff,
        length,
        tick_size,
        gain,
        loss,
    ):
        fixed_strategy = (
            self.set_ema(length)
            .set_tick_size(tick_size)
            .get_irb_signals(payoff)
            .calculate_irb_signals()
            .calculate_fixed_pl_results(gain, loss)
        ).df_fixed_results

        return fixed_strategy

    def calculate_expected_value(self):
        data_frame = self.df_results.query("Result != 0")[["Result"]].copy()

        gain = data_frame["Result"] > 0
        loss = data_frame["Result"] < 0

        data_frame["Gain_Count"] = np.where(gain, 1, 0)
        data_frame["Loss_Count"] = np.where(loss, 1, 0)

        data_frame["Gain_Count"] = data_frame["Gain_Count"].cumsum()
        data_frame["Loss_Count"] = data_frame["Loss_Count"].cumsum()

        query_gains = data_frame.query("Result > 0")["Result"]
        query_loss = data_frame.query("Result < 0")["Result"]

        data_frame["Mean_Gain"] = query_gains.expanding().mean()
        data_frame["Mean_Loss"] = query_loss.expanding().mean()

        data_frame["Mean_Gain"].fillna(method="ffill", inplace=True)
        data_frame["Mean_Loss"].fillna(method="ffill", inplace=True)

        data_frame["Total_Gain"] = (
            np.where(gain, data_frame["Result"], 0)
        .cumsum()
        )
        data_frame["Total_Loss"] = (
            np.where(loss, data_frame["Result"], 0)
            .cumsum()
        )

        total_trade = data_frame["Gain_Count"] + data_frame["Loss_Count"]
        win_rate = data_frame["Gain_Count"] / total_trade
        loss_rate = data_frame["Loss_Count"] / total_trade

        data_frame["Total_Trade"] = total_trade
        data_frame["Win_Rate"] = win_rate
        data_frame["Loss_Rate"] = loss_rate

        # expected mathematical calculation
        em_gain = data_frame["Mean_Gain"] * data_frame["Win_Rate"]
        em_loss = data_frame["Mean_Loss"] * data_frame["Loss_Rate"]
        data_frame["EM"] = em_gain - abs(em_loss)

        return data_frame
