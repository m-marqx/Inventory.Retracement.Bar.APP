import tkinter as tk
import pandas as pd

from view.plot_irb import Plot

from model.strategy.params.indicators_params import (
    EmaParams,
    MACDParams,
    CCIParams,
)
from model.strategy.params.strategy_params import (
    TrendParams,
    IrbParams,
    IndicatorsParams,
)

from model.strategy.strategy import BuilderStrategy
from model.strategy.indicators import BuilderSource

from .strategy_params import (
    EMAParamsGUI,
    MACDParamsGUI,
    CCIParamsGUI,
    IRBParamsGUI,
    TrendParamsGUI,
    IndicatorTrendParamsGUI,
)

from .data import GetDataGUI


class RunStrategy(GetDataGUI):
    def __init__(self, master):
        self.master = master
        super().__init__(self.master)

        self.run_strategy_button = tk.Button(
            self.master,
            text="Run Strategy",
            command=self.run_strategy,
        )

        self.run_strategy_button.grid(row=1, column=2)

        self.ema_parameter = EMAParamsGUI(self.master)
        self.macd_parameter = MACDParamsGUI(self.master)
        self.cci_parameter = CCIParamsGUI(self.master)

        self.irb_parameter = IRBParamsGUI(self.master)
        self.trend_parameter = TrendParamsGUI(self.master)
        self.indicator_trend_parameter = IndicatorTrendParamsGUI(self.master)

        self.source_df = pd.DataFrame()
        self.strategy = pd.DataFrame()

    # Create a class to run_strategy
    def run_strategy(self):  # está funcionando só não tá inserindo o texto.
        if self.trend_parameter.ema_var.get():
            self.ema_params = EmaParams(
                source_column=self.ema_parameter.ema_source_var.get(),
                length=int(self.ema_parameter.ema_length_entry.get()),
            )
        if self.trend_parameter.macd_var.get():
            self.macd_params = MACDParams(
                source_column=self.macd_parameter.macd_source_var.get(),
                fast_length=int(self.macd_parameter.macd_fast_length_entry.get()),
                slow_length=int(self.macd_parameter.macd_slow_length_entry.get()),
                signal_length=int(self.macd_parameter.macd_signal_length_entry.get()),
            )
        if self.trend_parameter.cci_var.get():
            self.cci_params = CCIParams(
                source_column=self.cci_parameter.cci_source_var.get(),
                length=int(self.cci_parameter.cci_length_entry.get()),
                ma_type=self.cci_parameter.cci_ma_type_var.get(),
                constant=float(self.cci_parameter.cci_constant_entry.get()),
            )

        self.irb_params = IrbParams(
            lowestlow=int(self.irb_parameter.irb_lowestlow_entry.get()),
            payoff=float(self.irb_parameter.irb_payoff_entry.get()),
            tick_size=float(self.irb_parameter.irb_tick_size_entry.get()),
            wick_percentage=float(self.irb_parameter.irb_wick_percentage_entry.get()),
        )
        self.trend_params = TrendParams(
            ema=self.trend_parameter.ema_var.get(),
            macd=self.trend_parameter.macd_var.get(),
            cci=self.trend_parameter.cci_var.get(),
        )
        self.indicators_params = IndicatorsParams(
            ema_column=self.indicator_trend_parameter.indicators_ema_column_var.get(),
            macd_histogram_trend_value=float(
                self.indicator_trend_parameter.indicators_macd_histogram_trend_value_entry.get()
            ),
            cci_trend_value=float(
                self.indicator_trend_parameter.indicators_cci_trend_value_label_entry.get()
            ),
        )

        self.source_df = BuilderSource(self.df)

        if self.trend_parameter.ema_var.get():
            self.source_df = self.source_df.set_EMA_params(self.ema_params).set_ema()
        if self.trend_parameter.macd_var.get():
            self.source_df = self.source_df.set_MACD_params(self.macd_params).set_macd()
        if self.trend_parameter.cci_var.get():
            self.source_df = self.source_df.set_CCI_params(self.cci_params).set_cci()
        self.source_df = self.source_df.execute()

        self.strategy = (
            BuilderStrategy(
                self.source_df,
            )
            .set_trend_params(self.indicators_params, self.trend_params)
            .get_trend()
            .set_irb_params(self.irb_params)
            .get_irb_signals()
            .calculate_irb_signals()
            .calculateResults()
            .execute()
        )

        self.insert_text("Strategy executed successfully.\n")


class StrategyButton(RunStrategy):
    def __init__(self, master):
        self.master = master
        super().__init__(self.master)

        self.cols = ["open", "high", "low", "close", "Result", "Cumulative_Result"]
        self.strategy_check = self.strategy.columns.equals(self.cols)

        self.winrate_button = tk.Button(
            self.master,
            text="Win Rate",
            command=lambda: [
                Plot(self.strategy)
                .winrate()
                .fig_to_html(title="Win Rate", open_file=True),
                self.strategy_check,
            ],
        )
        self.winrate_button.grid(row=3, column=0)

        self.results_button = tk.Button(
            self.master,
            text="Results",
            command=lambda: [
                Plot(self.strategy)
                .results()
                .fig_to_html(title="Result", open_file=True),
                self.strategy_check,
            ],
        )
        self.results_button.grid(row=3, column=1)

        self.chart_button = tk.Button(
            self.master,
            text="Chart",
            command=lambda: [
                Plot(self.strategy).chart().fig_to_html(title="Chart", open_file=True),
                self.strategy_check,
            ],
        )
        self.chart_button.grid(row=3, column=2)

        self.trading_results_button = tk.Button(
            self.master,
            text="Trading Results",
            command=lambda: [
                Plot(self.strategy)
                .trading_results()
                .fig_to_html(title="Trading Results", open_file=True),
                self.strategy_check,
            ],
        )
        self.trading_results_button.grid(row=3, column=3)
