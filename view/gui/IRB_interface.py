import tkinter as tk
from tkinter import ttk

import pandas as pd

from controller.future_API import FuturesAPI
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
from view.plot_irb import Plot

from .labels import Label, Grid
from .data import GetDataGUI


class Interface:
    def __init__(self, master):
        self.master = master
        self.master.title("Futures Trading Strategy")

        self.label = Label(self.master)
        self.grid = Grid(self.master)
        self.get_data = GetDataGUI(self.master)

        self.run_strategy_button = tk.Button(
            master, text="Run Strategy", command=self.run_strategy
        )
        self.run_strategy_button.grid(row=1, column=2)

        self.strategy = pd.DataFrame()
        # Verifique o número de colunas do dataframe
        self.cols = [
            "open",
            "high",
            "low",
            "close",
            "ema",
            "MACD_Histogram",
            "CCI",
            "uptrend",
            "IRB_Condition",
            "Signal",
            "Entry_Price",
            "Take_Profit",
            "Stop_Loss",
            "Close_Position",
            "Signal_Condition",
            "Result",
            "order_fill_price",
            "Cumulative_Result",
        ]
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

        # set default values for the input parameters
        # EMA Parameters
        self.options = ["open", "high", "low", "close"]

        self.ema_source_var = tk.StringVar()
        self.ema_source_entry = tk.OptionMenu(
            master, self.ema_source_var, *self.options
        )
        self.ema_source_entry.grid(row=5, column=1)

        self.ema_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.ema_length_entry.grid(row=5, column=3)

        # MACD Parameters
        self.macd_source_var = tk.StringVar()
        self.macd_source_entry = tk.OptionMenu(
            master, self.macd_source_var, *self.options
        )
        self.macd_source_entry.grid(row=7, column=1)

        self.macd_fast_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.macd_fast_length_entry.grid(row=7, column=3)

        self.macd_slow_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.macd_slow_length_entry.grid(row=8, column=1)

        self.macd_signal_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.macd_signal_length_entry.grid(row=8, column=3)

        # CCI Parameters
        self.cci_source_var = tk.StringVar()
        self.cci_source_entry = tk.OptionMenu(
            master, self.cci_source_var, *self.options
        )
        self.cci_source_entry.grid(row=10, column=1)

        self.cci_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.cci_length_entry.grid(row=10, column=3)

        self.ma_options = ["sma", "ema"]
        self.cci_ma_type_var = tk.StringVar()
        self.cci_ma_type_entry = tk.OptionMenu(
            master, self.cci_ma_type_var, *self.ma_options
        )
        self.cci_ma_type_entry.grid(row=11, column=1)

        self.cci_constant_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.cci_constant_entry.grid(row=11, column=3)

        # IRB Parameters
        self.irb_lowestlow_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10, value=1
        )
        self.irb_lowestlow_entry.grid(row=13, column=1)

        self.irb_payoff_entry = ttk.Spinbox(
            master, width=10, increment=0.1, from_=0, to=1e10, value=2
        )
        self.irb_payoff_entry.grid(row=13, column=3)

        self.irb_tick_size_entry = ttk.Spinbox(
            master, width=10, increment=0.1, from_=0, to=1e10, value=0.1
        )
        self.irb_tick_size_entry.grid(row=14, column=1)

        self.irb_wick_percentage_entry = ttk.Spinbox(
            master, width=10, increment=0.01, from_=0, to=1, value=0.02
        )
        self.irb_wick_percentage_entry.grid(row=14, column=3)

        self.ema_var = tk.BooleanVar()
        self.trend_ema_entry = tk.Checkbutton(master, width=10, variable=self.ema_var)
        self.trend_ema_entry.grid(row=16, column=1)

        self.cci_var = tk.BooleanVar()
        self.trend_cci_entry = tk.Checkbutton(master, width=10, variable=self.cci_var)
        self.trend_cci_entry.grid(row=16, column=3)

        self.macd_var = tk.BooleanVar()
        self.trend_macd_entry = tk.Checkbutton(master, width=10, variable=self.macd_var)
        self.trend_macd_entry.grid(row=17, column=1)

        self.indicators_ema_column_var = tk.StringVar()
        self.indicators_ema_column_entry = tk.OptionMenu(
            master, self.indicators_ema_column_var, *self.options
        )
        self.indicators_ema_column_entry.grid(row=19, column=1)

        self.indicators_macd_histogram_trend_value_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10, value=0
        )
        self.indicators_macd_histogram_trend_value_entry.grid(row=19, column=3)

        self.indicators_cci_trend_value_label_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10, value=0
        )
        self.indicators_cci_trend_value_label_entry.grid(row=20, column=1)

    def run_strategy(self):
        if self.ema_var.get():
            self.ema_params = EmaParams(
                source_column=self.ema_source_var.get(),
                length=int(self.ema_length_entry.get()),
            )
        if self.macd_var.get():
            self.macd_params = MACDParams(
                source_column=self.macd_source_var.get(),
                fast_length=int(self.macd_fast_length_entry.get()),
                slow_length=int(self.macd_slow_length_entry.get()),
                signal_length=int(self.macd_signal_length_entry.get()),
            )
        if self.cci_var.get():
            self.cci_params = CCIParams(
                source_column=self.cci_source_var.get(),
                length=int(self.cci_length_entry.get()),
                ma_type=self.cci_ma_type_var.get(),
                constant=float(self.cci_constant_entry.get()),
            )

        self.irb_params = IrbParams(
            lowestlow=int(self.irb_lowestlow_entry.get()),
            payoff=float(self.irb_payoff_entry.get()),
            tick_size=float(self.irb_tick_size_entry.get()),
            wick_percentage=float(self.irb_wick_percentage_entry.get()),
        )
        self.trend_params = TrendParams(
            ema=self.ema_var.get(),
            macd=self.macd_var.get(),
            cci=self.cci_var.get(),
        )
        self.indicators_params = IndicatorsParams(
            ema_column=self.indicators_ema_column_var.get(),
            macd_histogram_trend_value=float(
                self.indicators_macd_histogram_trend_value_entry.get()
            ),
            cci_trend_value=float(self.indicators_cci_trend_value_label_entry.get()),
        )

        try:
            self.source_df = BuilderSource(self.get_data.df)

            if self.ema_var.get():
                self.source_df = self.source_df.set_EMA_params(
                    self.ema_params
                ).set_ema()
            if self.macd_var.get():
                self.source_df = self.source_df.set_MACD_params(
                    self.macd_params
                ).set_macd()
            if self.cci_var.get():
                self.source_df = self.source_df.set_CCI_params(
                    self.cci_params
                ).set_cci()
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

            self.get_data.insert_text("Strategy executed successfully.\n")
        except Exception as e:
            self.get_data.insert_text(f"Error occurred: {str(e)}\n")


# %%
