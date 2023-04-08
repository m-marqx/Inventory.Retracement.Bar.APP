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


class InventoryRetracementBarGUI:
    def __init__(self, master):
        self.master = master
        master.title("Futures Trading Strategy")

        self.label1 = tk.Label(master, text="Symbol:")
        self.label1.grid(row=0, column=0)

        self.symbol_entry = tk.Entry(master, width=10)
        self.symbol_entry.grid(row=0, column=1)

        self.label2 = tk.Label(master, text="Timeframe:")
        self.label2.grid(row=0, column=2)

        self.timeframe_entry = tk.Entry(master, width=10)
        self.timeframe_entry.grid(row=0, column=3)

        self.get_data_button = tk.Button(master, text="Get Data", command=self.get_data)
        self.get_data_button.grid(row=1, column=1)

        self.run_strategy_button = tk.Button(
            master, text="Run Strategy", command=self.run_strategy
        )
        self.run_strategy_button.grid(row=1, column=2)

        self.text_widget = tk.Text(master, height=8, width=80)
        self.text_widget.grid(row=2, column=0, columnspan=4)
        self.text_widget.configure(state="disabled")

        # self.df = pd.read_parquet('BTCPERP.parquet')
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

        self.ema_params_label = tk.Label(master, text="EMA", bg="#333333", fg="#FFFFFF")
        self.ema_params_label.grid(row=4, column=0, columnspan=4, sticky="NSWE")

        self.label_ema_source = tk.Label(master, text="EMA Source Column:")
        self.label_ema_source.grid(row=5, column=0)

        self.ema_source_var = tk.StringVar()
        self.ema_source_entry = tk.OptionMenu(
            master, self.ema_source_var, *self.options
        )
        self.ema_source_entry.grid(row=5, column=1)

        self.label_ema_length = tk.Label(master, text="EMA Length:")
        self.label_ema_length.grid(row=5, column=2)

        self.ema_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.ema_length_entry.grid(row=5, column=3)

        # MACD Parameters
        self.macd_params_label = tk.Label(
            master, text="MACD", bg="#333333", fg="#FFFFFF"
        )
        self.macd_params_label.grid(row=6, column=0, columnspan=4, sticky="NSWE")

        self.label_macd_source = tk.Label(master, text="MACD Source Column:")
        self.label_macd_source.grid(row=7, column=0)

        self.macd_source_var = tk.StringVar()
        self.macd_source_entry = tk.OptionMenu(
            master, self.macd_source_var, *self.options
        )
        self.macd_source_entry.grid(row=7, column=1)

        self.label_macd_fast_length = tk.Label(master, text="MACD Fast Length:")
        self.label_macd_fast_length.grid(row=7, column=2)

        self.macd_fast_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.macd_fast_length_entry.grid(row=7, column=3)

        self.label_macd_slow_length = tk.Label(master, text="MACD Slow Length:")
        self.label_macd_slow_length.grid(row=8, column=0)

        self.macd_slow_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.macd_slow_length_entry.grid(row=8, column=1)

        self.label_macd_signal_length = tk.Label(master, text="MACD Signal Length:")
        self.label_macd_signal_length.grid(row=8, column=2)

        self.macd_signal_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.macd_signal_length_entry.grid(row=8, column=3)

        # CCI Parameters
        self.cci_params_label = tk.Label(master, text="CCI", bg="#333333", fg="#FFFFFF")
        self.cci_params_label.grid(row=9, column=0, columnspan=4, sticky="NSWE")

        self.label_cci_source = tk.Label(master, text="CCI Source Column:")
        self.label_cci_source.grid(row=10, column=0)

        self.cci_source_var = tk.StringVar()
        self.cci_source_entry = tk.OptionMenu(
            master, self.cci_source_var, *self.options
        )
        self.cci_source_entry.grid(row=10, column=1)

        self.label_cci_length = tk.Label(master, text="CCI Length:")
        self.label_cci_length.grid(row=10, column=2)

        self.cci_length_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.cci_length_entry.grid(row=10, column=3)

        self.label_cci_ma_type = tk.Label(master, text="CCI MA Type:")
        self.label_cci_ma_type.grid(row=11, column=0)

        self.ma_options = ["sma", "ema"]
        self.cci_ma_type_var = tk.StringVar()
        self.cci_ma_type_entry = tk.OptionMenu(
            master, self.cci_ma_type_var, *self.ma_options
        )
        self.cci_ma_type_entry.grid(row=11, column=1)

        self.label_cci_constant = tk.Label(master, text="CCI Constant:")
        self.label_cci_constant.grid(row=11, column=2)

        self.cci_constant_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10
        )
        self.cci_constant_entry.grid(row=11, column=3)

        # IRB Parameters
        self.irb_params_label = tk.Label(
            master, text="IRB Params", bg="#333333", fg="#FFFFFF"
        )
        self.irb_params_label.grid(row=12, column=0, columnspan=4, sticky="NSWE")

        self.irb_lowestlow_label = tk.Label(master, text="Lowest Low:")
        self.irb_lowestlow_label.grid(row=13, column=0)

        self.irb_lowestlow_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10, value=1
        )
        self.irb_lowestlow_entry.grid(row=13, column=1)

        self.irb_payoff_label = tk.Label(master, text="Payoff:")
        self.irb_payoff_label.grid(row=13, column=2)

        self.irb_payoff_entry = ttk.Spinbox(
            master, width=10, increment=0.1, from_=0, to=1e10, value=2
        )
        self.irb_payoff_entry.grid(row=13, column=3)

        self.irb_tick_size_label = tk.Label(master, text="Tick Size:")
        self.irb_tick_size_label.grid(row=14, column=0)

        self.irb_tick_size_entry = ttk.Spinbox(
            master, width=10, increment=0.1, from_=0, to=1e10, value=0.1
        )
        self.irb_tick_size_entry.grid(row=14, column=1)

        self.irb_wick_percentage_label = tk.Label(master, text="Wick Percentage:")
        self.irb_wick_percentage_label.grid(row=14, column=2)

        self.irb_wick_percentage_entry = ttk.Spinbox(
            master, width=10, increment=0.01, from_=0, to=1, value=0.02
        )
        self.irb_wick_percentage_entry.grid(row=14, column=3)

        self.trend_params_label = tk.Label(
            master, text="Trend Params", bg="#333333", fg="#FFFFFF"
        )
        self.trend_params_label.grid(row=15, column=0, columnspan=4, sticky="NSWE")

        self.trend_ema_label = tk.Label(master, text="EMA:")
        self.trend_ema_label.grid(row=16, column=0)

        self.ema_var = tk.BooleanVar()
        self.trend_ema_entry = tk.Checkbutton(master, width=10, variable=self.ema_var)
        self.trend_ema_entry.grid(row=16, column=1)

        self.trend_cci_label = tk.Label(master, text="CCI:")
        self.trend_cci_label.grid(row=16, column=2)

        self.cci_var = tk.BooleanVar()
        self.trend_cci_entry = tk.Checkbutton(master, width=10, variable=self.cci_var)
        self.trend_cci_entry.grid(row=16, column=3)

        self.trend_macd_label = tk.Label(master, text="MACD:")
        self.trend_macd_label.grid(row=17, column=0)

        self.macd_var = tk.BooleanVar()
        self.trend_macd_entry = tk.Checkbutton(master, width=10, variable=self.macd_var)
        self.trend_macd_entry.grid(row=17, column=1)

        self.indicators_params_label = tk.Label(
            master, text="Indicators Params", bg="#333333", fg="#FFFFFF"
        )
        self.indicators_params_label.grid(row=18, column=0, columnspan=4, sticky="NSWE")

        self.indicators_ema_column_label = tk.Label(master, text="EMA Column:")
        self.indicators_ema_column_label.grid(row=19, column=0)

        self.indicators_ema_column_var = tk.StringVar()
        self.indicators_ema_column_entry = tk.OptionMenu(
            master, self.indicators_ema_column_var, *self.options
        )
        self.indicators_ema_column_entry.grid(row=19, column=1)

        self.indicators_macd_histogram_trend_value_label = tk.Label(
            master, text="MACD Histogram Trend Value:"
        )
        self.indicators_macd_histogram_trend_value_label.grid(row=19, column=2)

        self.indicators_macd_histogram_trend_value_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10, value=0
        )
        self.indicators_macd_histogram_trend_value_entry.grid(row=19, column=3)

        self.indicators_cci_trend_value_label = tk.Label(
            master, text="CCI Trend Value:"
        )
        self.indicators_cci_trend_value_label.grid(row=20, column=0)

        self.indicators_cci_trend_value_label_entry = ttk.Spinbox(
            master, width=10, increment=1, from_=0, to=1e10, value=0
        )
        self.indicators_cci_trend_value_label_entry.grid(row=20, column=1)

    def insert_text(self, text):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, f"{text}\n")
        self.text_widget.configure(state="disabled")

    def get_data(self):
        symbol = self.symbol_entry.get()
        interval = self.timeframe_entry.get()

        try:
            # Change button text to "loading"
            self.get_data_button.config(state="disabled", text="Loading...")
            self.master.update()

            FAPI = FuturesAPI()
            self.df = FAPI.get_all_futures_klines_df(symbol, interval)

            self.insert_text(
                f"Retrieved {len(self.df)} klines for {symbol} with interval {interval}\n"
            )

        except Exception as e:
            # Handle the exception, for example, show an error message in the text widget
            self.insert_text(f"Error occurred: {str(e)}\n")

        finally:
            # Change button text back to "Get Data"
            self.get_data_button.config(state="normal", text="Get Data")
            self.master.update()

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
            self.source_df = BuilderSource(self.df)

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

            self.insert_text("Strategy executed successfully.\n")
        except Exception as e:
            self.insert_text(f"Error occurred: {str(e)}\n")

