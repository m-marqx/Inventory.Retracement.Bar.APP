import pandas as pd
import numpy as np
from model.strategy.params.strategy_params import (
    irb_params,
    trend_params,
    indicators_params,
)
from model.utils import BaseStrategy

class calculate_trend(BaseStrategy):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.conditions = pd.DataFrame()

    def ema_condition(self, source_column: str):
        if "ema" in self.df_filtered.columns:
            self.conditions["ema"] = (
                self.df_filtered[source_column] > self.df_filtered["ema"]
            )
        else:
            raise ValueError("EMA column not found")
        return self

    def macd_condition(self, trend_value: int):
        if "MACD_Histogram" in self.df_filtered.columns:
            self.conditions["MACD_Histogram"] = (
                self.df_filtered["MACD_Histogram"]
                > trend_value
            )

        else:
            raise ValueError("MACD_Histogram column not found")
        return self

    def cci_condition(self, trend_value: int):
        if "CCI" in self.df_filtered.columns:
            self.conditions["CCI"] = (self.df_filtered["CCI"]
            > trend_value
        )

        else:
            raise ValueError("CCI column not found")
        return self

    def execute(self):
        self.conditions["uptrend"] = self.conditions.all(axis=1)
        self.df_filtered["uptrend"] = self.conditions["uptrend"]
        self.df_filtered["uptrend"].fillna(True, inplace=True)
        return self.df_filtered


class set_trend(BaseStrategy):
    def __init__(self, dataframe, params: indicators_params):
        super().__init__(dataframe)
        self.params = params
        self.trend_params = trend_params()

    def execute(self):
        self.trend = calculate_trend(self.df_filtered)

        if self.trend_params.ema:
            self.trend = self.trend.ema_condition(self.params.ema_column)

        if self.trend_params.cci:
            self.trend = self.trend.cci_condition(self.params.cci_trend_value)

        if self.trend_params.macd:
            self.trend = self.trend.macd_condition(
                self.params.macd_histogram_trend_value
            )
        self.params.trend = True
        return self


# %%
class set_ticksize(BaseStrategy):
    def __init__(self, tick_size=0.1):
        self.tick_size = tick_size

    def execute(self):
        return self.tick_size


class GetIrbSignals_buy(BaseStrategy):
    def __init__(self, dataframe, params: irb_params):
        super().__init__(dataframe)
        self.lowestlow = params.lowestlow
        self.payoff = params.payoff
        self.tick_size = params.tick_size
        self.wick_percentage = params.wick_percentage

        self.open_price = self.df_filtered["open"]
        self.high_price = self.df_filtered["high"]
        self.low_price = self.df_filtered["low"]
        self.close_price = self.df_filtered["close"]

    def execute(self):
        self.candle_amplitude = self.high_price - self.low_price
        self.candle_downtail = (
            np.minimum(self.open_price, self.close_price) - self.low_price
        )
        self.candle_uppertail = self.high_price - np.maximum(
            self.open_price, self.close_price
        )

        # Analyze the downtail and uptail of the candle,
        # and assign a value to the column IRB_Condition
        # based on the proportion of the wick.
        self.bullish_calculation = self.candle_uppertail / self.candle_amplitude
        self.bearish_calculation = self.candle_downtail / self.candle_amplitude

        self.df_filtered["IRB_Condition"] = np.where(
            self.df_filtered["uptrend"],
            self.bullish_calculation,
            self.bearish_calculation,
        )

        self.irb_condition = self.df_filtered["IRB_Condition"] >= self.wick_percentage
        self.buy_condition = self.irb_condition & self.df_filtered["uptrend"]

        self.df_filtered["Signal"] = np.where(self.buy_condition, 1, np.nan)
        self.df_filtered["Signal"].astype("float32")

        self.entry_price = self.df_filtered["high"].shift(1) + self.tick_size
        self.target = self.df_filtered["high"].shift(1) + (
            self.candle_amplitude.shift(1)
            * self.payoff
        )

        # Stop Loss is the lowest low of the last X candles
        self.stop_loss = (
            self.df_filtered["low"].rolling(self.lowestlow).min().shift()
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

        return self.df_filtered


class CalculateIrbSignals(BaseStrategy):
    def __init__(self, dataframe):
        super().__init__(dataframe)

    def execute(self):
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
        self.df_filtered["Close_Position"] = self.close_position_arr

        return self.df_filtered


class CheckIrbSignals(BaseStrategy):
    def __init__(self, dataframe):
        super().__init__(dataframe)

    def execute(self):
        columns = ["Signal", "Close_Position"]
        self.df_check = self.df_filtered[columns].copy()

        self.df_check["Signal_Shifted"] = self.df_check["Signal"].shift(1)

        is_null_signal = self.df_check["Signal"].isnull()
        is_null_signal_shift = self.df_check["Signal_Shifted"].isnull()

        has_signal_error = (
            is_null_signal
            & (self.df_check["Signal_Shifted"] == 1)
        )

        has_close_error = is_null_signal_shift & self.df_check["Close_Position"]

        has_error = has_signal_error | has_close_error

        if has_error.any():
            print("Error Found")

        self.df_check["Error"] = has_error
        return self


class calculateResults(BaseStrategy):
    def __init__(self, dataframe):
        super().__init__(dataframe)

    def execute(self, verify_error=True):
        self.is_close_position = self.df_filtered["Close_Position"]
        self.is_take_profit = (
            self.df_filtered["high"]
            > self.df_filtered["Take_Profit"]
        )

        self.is_stop_loss = (
            self.df_filtered["low"]
            < self.df_filtered["Stop_Loss"]
        )

        self.profit = (
            self.df_filtered["Take_Profit"]
            - self.df_filtered["Entry_Price"]
        )
        self.loss = (
            self.df_filtered["Stop_Loss"]
            - self.df_filtered["Entry_Price"]
        )

        self.df_filtered["Result"] = 0
        self.df_filtered["Result"] = np.where(
            self.is_close_position & self.is_take_profit,
            self.profit,
            self.df_filtered["Result"],
        )

        self.df_filtered["Result"] = np.where(
            self.is_close_position & self.is_stop_loss,
            self.loss,
            self.df_filtered["Result"],
        )

        self.df_filtered["Cumulative_Result"] = (
            self.df_filtered["Result"]
            .cumsum()
        )

        if verify_error:
            CheckIrbSignals(self.df_filtered).execute()

        return self.df_filtered


class builder_strategy(BaseStrategy):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.trend = calculate_trend(self.df_filtered)

    def set_trend_params(self, params: indicators_params):
        #! The copy is necessary to avoid the indicators_params
        #! being changed when the indicators are setted
        self.indicators_params = params.copy()
        return self

    def set_trend(self):
        self.trend = calculate_trend(self.df_filtered)

        if "ema" in self.df_filtered.columns:
            self.trend = (
                self.trend.ema_condition(self.indicators_params.ema_column)
            )

        if "CCI" in self.df_filtered.columns:
            self.trend = self.trend.cci_condition(
                self.indicators_params.cci_trend_value
            )

        if "MACD_Histogram" in self.df_filtered.columns:
            self.trend = self.trend.macd_condition(
                self.indicators_params.macd_histogram_trend_value
            )
        self.trend.execute()
        return self

    def set_irb_params(self, params: irb_params):
        self.irb_params = params
        return self

    def get_irb_signals(self):
        self.df_filtered = GetIrbSignals_buy(
            self.df_filtered, self.irb_params
        ).execute()
        return self

    def calculate_irb_signals(self):
        self.df_filtered = CalculateIrbSignals(self.df_filtered).execute()
        return self

    def calculateResults(self):
        self.df_filtered = calculateResults(self.df_filtered).execute()
        return self

    def execute(self):
        return self.df_filtered
