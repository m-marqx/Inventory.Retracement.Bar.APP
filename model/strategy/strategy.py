import pandas as pd
import numpy as np
from model.strategy.params import (
    IrbParams,
    TrendParams,
    IndicatorsParams,
    ResultParams,
)

from model.utils import (
    BaseStrategy,
    BrokerEmulator,
    CalculateTradePerformance,
)


class CalculateTrend(BaseStrategy):
    def __init__(self, dataframe):
        """
        Initialize the CalculateTrend object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        """
        super().__init__(dataframe)
        self.conditions = pd.DataFrame()

    def ema_condition(self, source_column: str):
        """
        Set the EMA condition for trend calculation.

        Parameters:
        -----------
        source_column : str
            The source column name.

        Returns:
        --------
        CalculateTrend
            The CalculateTrend object.
        """
        if "ema" in self.df_filtered.columns:
            self.conditions["ema"] = (
                self.df_filtered[source_column] > self.df_filtered["ema"]
            )
        else:
            raise ValueError("EMA column not found")
        return self

    def macd_condition(self, trend_value: int):
        """
        Set the MACD condition for trend calculation.

        Parameters:
        -----------
        trend_value : int
            The trend value for MACD calculation.

        Returns:
        --------
        CalculateTrend
            The CalculateTrend object.
        """
        if "MACD_Histogram" in self.df_filtered.columns:
            self.conditions["MACD_Histogram"] = (
                self.df_filtered["MACD_Histogram"]
                > trend_value
            )

        else:
            raise ValueError("MACD_Histogram column not found")
        return self

    def cci_condition(self, trend_value: int):
        """
        Set the CCI condition for trend calculation.

        Parameters:
        -----------
        trend_value : int
            The trend value for CCI calculation.

        Returns:
        --------
        CalculateTrend
            The CalculateTrend object.
        """
        if "CCI" in self.df_filtered.columns:
            self.conditions["CCI"] = self.df_filtered["CCI"] > trend_value

        else:
            raise ValueError("CCI column not found")
        return self

    def execute(self):
        """
        The function executes a filtering process on a DataFrame and adds a new
        column indicating whether the data is in an uptrend or not.

        Returns:
        --------
            pd.DataFrame
                The function `execute` returns the filtered DataFrame with an
                additional `uptrend` column that contains boolean values indicating
                whether the conditions for an uptrend are met or not. If no indicator
                is set, the `uptrend` column for all rows will be `True`.
        """
        self.conditions["uptrend"] = self.conditions.all(axis=1)
        self.df_filtered["uptrend"] = self.conditions["uptrend"]
        self.df_filtered["uptrend"].fillna(True, inplace=True)
        return self.df_filtered


class SetTrend(BaseStrategy):
    def __init__(self, dataframe, params: IndicatorsParams, trend_params: TrendParams):
        """
        Initialize the SetTrend object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        params : IndicatorsParams
            The parameters for indicators.
        trend_params : TrendParams
            The parameters for trend calculation.
        """
        super().__init__(dataframe)
        self.params = params
        self.trend_params = trend_params.copy()

    def execute(self):
        """
        Execute the setting of trends.

        Returns:
        --------
        SetTrend
            The SetTrend object.
        """
        self.trend = CalculateTrend(self.df_filtered)

        if self.trend_params.ema:
            self.trend = self.trend.ema_condition(self.params.ema_column)

        if self.trend_params.cci:
            self.trend = self.trend.cci_condition(self.params.cci_trend_value)

        if self.trend_params.macd:
            self.trend = self.trend.macd_condition(
                self.params.macd_histogram_trend_value
            )

        self.trend.execute()
        return self


class SetTicksize(BaseStrategy):
    def __init__(self, tick_size=0.1):
        """
        Initialize the SetTicksize object.

        Parameters:
        -----------
        tick_size : float, optional
            The tick size, by default 0.1.
        """
        self.tick_size = tick_size

    def execute(self):
        """
        Execute the tick size setting.

        Returns:
        --------
        float
            The tick size.
        """
        return self.tick_size


class GetIrbSignalsBuy(BaseStrategy):
    def __init__(self, dataframe, params: IrbParams):
        """
        Initialize the GetIrbSignalsBuy object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        params : IrbParams
            The parameters for IRB signals.
        """
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
        """
        Execute the calculation of IRB signals.

        Returns:
        --------
        pd.DataFrame
            The processed dataframe with IRB signals.
        """
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
        """
        Initialize the CalculateIrbSignals object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        """
        super().__init__(dataframe)
        self.signal_arr = self.df_filtered["Signal"].to_numpy()
        self.entry_price_arr = self.df_filtered["Entry_Price"].to_numpy()
        self.take_profit_arr = self.df_filtered["Take_Profit"].to_numpy()
        self.stop_loss_arr = self.df_filtered["Stop_Loss"].to_numpy()
        self.high_price_arr = self.df_filtered["high"].to_numpy()
        self.low_price_arr = self.df_filtered["low"].to_numpy()
        self.close_position_arr = np.zeros(len(self.df_filtered), dtype=bool)

        self.signal_condition = self.signal_arr == 1
        self.profit = np.nan
        self.loss = np.nan

    def execute(self):
        """
        Execute the calculation of IRB signals.

        Returns:
        --------
        pd.DataFrame
            The processed dataframe with calculated signals.
        """
        for index in range(1, len(self.df_filtered)):
            prev_index = index - 1
            self.signal_condition[index] = self.signal_arr[prev_index] == 1
            if self.close_position_arr[prev_index]:
                self.signal_arr[index] = np.nan
                self.entry_price_arr[index] = np.nan
                self.take_profit_arr[index] = np.nan
                self.stop_loss_arr[index] = np.nan

            if self.signal_condition[index]:
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

                if self.profit | self.loss:
                    self.close_position_arr[index] = True
                    self.signal_arr[index] = -1

                elif self.profit & self.loss:
                    self.close_position_arr[index] = True
                    self.signal_arr[index] = -2

        self.df_filtered["Signal"] = self.signal_arr
        self.df_filtered["Signal"].fillna(0, inplace=True)

        self.df_filtered["Position"] = np.where(
            self.df_filtered["Signal"] != self.df_filtered["Signal"].shift(),
            self.df_filtered["Signal"],
            0,
        ).astype("int8")

        self.df_filtered["Entry_Price"] = self.entry_price_arr
        self.df_filtered["Take_Profit"] = self.take_profit_arr
        self.df_filtered["Stop_Loss"] = self.stop_loss_arr
        self.df_filtered["Close_Position"] = self.close_position_arr
        self.df_filtered["Signal_Condition"] = self.signal_condition

        return self.df_filtered


class CheckIrbSignals(BaseStrategy):
    def __init__(self, dataframe):
        """
        Initialize the CheckIrbSignals object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        """
        super().__init__(dataframe)

    def execute(self):
        """
        Execute the checking of IRB signals.

        Returns:
        --------
        CheckIrbSignals
            The CheckIrbSignals object.
        """
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


class CalculateResults(BaseStrategy):
    def __init__(self, dataframe: pd.DataFrame, params: ResultParams):
        """
        Initialize the CalculateResults object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.

        params : ResultParams
            An instance of the ResultParams class containing various parameters
            related to the computation.

        """
        super().__init__(dataframe)
        self.params = params

    def execute(self, verify_error=True):
        """
        Execute the calculation of results.

        Parameters:
        -----------
        verify_error : bool, optional
            Whether to verify for errors in the calculation (default is True).

        Returns:
        --------
        pd.DataFrame
            The processed dataframe with calculated results.
        """
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

        #Update Result column with broker emulator
        BrokerEmulator(self.df_filtered).broker_emulator_result()

        self.df_filtered["Cumulative_Result"] = (
            self.df_filtered["Result"]
            .cumsum()
        )

        capital = self.params.capital
        percent = self.params.percent
        gain = self.params.gain
        loss = self.params.loss
        method = self.params.method
        qty = self.params.qty
        coin_margined = self.params.coin_margined

        ctp = CalculateTradePerformance(self.df_filtered, capital, percent)
        if method == "Fixed":
            self.df_filtered = ctp.fixed(gain, loss)
        else:
            self.df_filtered = ctp.normal(qty, coin_margined)

        if verify_error:
            CheckIrbSignals(self.df_filtered).execute()

        return self.df_filtered


class BuilderStrategy(BaseStrategy):
    def __init__(self, dataframe):
        """
        Initialize the BuilderStrategy object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe.
        """
        super().__init__(dataframe)

    def set_trend_params(self, params: IndicatorsParams = IndicatorsParams(), trend_params: TrendParams = TrendParams()):
        """
        Set the parameters for trend analysis.

        Parameters:
        -----------
        params : IndicatorsParams, optional
            The parameters for indicators (default is IndicatorsParams()).
        trend_params : TrendParams, optional
            The parameters for trend analysis (default is TrendParams()).

        Returns:
        --------
        BuilderStrategy
            The BuilderStrategy object.
        """
        self.indicators_params = params
        self.trend_params = trend_params
        return self

    def get_trend(self):
        """
        Get the trend analysis.

        Returns:
        --------
        BuilderStrategy
            The BuilderStrategy object.
        """
        self.trend = SetTrend(self.df_filtered, self.indicators_params, self.trend_params).execute()
        return self

    def set_irb_params(self, params: IrbParams = IrbParams()):
        """
        Set the parameters for IRB signals.

        Parameters:
        -----------
        params : IrbParams, optional
            The parameters for IRB signals (default is IrbParams()).

        Returns:
        --------
        BuilderStrategy
            The BuilderStrategy object.
        """
        self.irb_params = params
        return self

    def get_irb_signals(self):
        """
        Get the IRB signals.

        Returns:
        --------
        BuilderStrategy
            The BuilderStrategy object.
        """
        self.df_filtered = GetIrbSignalsBuy(self.df_filtered, self.irb_params).execute()
        return self

    def calculate_irb_signals(self):
        """
        Calculate the IRB signals.

        Returns:
        --------
        BuilderStrategy
            The BuilderStrategy object.
        """
        self.df_filtered = CalculateIrbSignals(self.df_filtered).execute()
        return self

    def set_result_params(self, params: ResultParams = ResultParams()):
        """
        Set the result parameters for an object.

        Parameters:
        -----------
        params : ResultParams
            An instance of the ResultParams that contains various parameters
            related to the result of a computation. If `params` is not provided,
            the default value is an instance of ResultParams.

        Returns:
        --------
        self
            The object itself.
        """
        self.result_params = params
        return self

    def calculateResults(self):
        """
        Calculate the results.

        Returns:
        --------
        BuilderStrategy
            The BuilderStrategy object.
        """
        self.df_filtered = CalculateResults(self.df_filtered, self.result_params).execute()
        return self

    def execute(self):
        """
        Execute the strategy.

        Returns:
        --------
        pd.DataFrame
            The processed dataframe.
        """
        return self.df_filtered
