import tkinter as tk
from tkinter import ttk


class EMAParamsGUI:
    def __init__(self, master):
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


class MACDParamsGUI:
    def __init__(self, master):
        self.options = ["open", "high", "low", "close"]
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


class CCIParamsGUI:
    def __init__(self, master):
        self.options = ["open", "high", "low", "close"]
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


class IRBParamsGUI:
    def __init__(self, master):
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


class TrendParamsGUI:
    def __init__(self, master):
        self.ema_var = tk.BooleanVar()
        self.trend_ema_entry = tk.Checkbutton(master, width=10, variable=self.ema_var)
        self.trend_ema_entry.grid(row=16, column=1)

        self.cci_var = tk.BooleanVar()
        self.trend_cci_entry = tk.Checkbutton(master, width=10, variable=self.cci_var)
        self.trend_cci_entry.grid(row=16, column=3)

        self.macd_var = tk.BooleanVar()
        self.trend_macd_entry = tk.Checkbutton(master, width=10, variable=self.macd_var)
        self.trend_macd_entry.grid(row=17, column=1)


class IndicatorTrendParamsGUI:
    def __init__(self, master):
        self.options = ["open", "high", "low", "close"]
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
