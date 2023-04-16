import tkinter as tk


class Label:
    def __init__(self, master):
        self.label1 = tk.Label(
            master,
            text="Symbol:",
        )
        self.label2 = tk.Label(
            master,
            text="Timeframe:",
        )

        self.ema_params_label = tk.Label(
            master,
            text="EMA",
            bg="#333333",
            fg="#FFFFFF",
        )

        self.label_ema_source = tk.Label(
            master,
            text="EMA Source Column:",
        )

        self.label_ema_length = tk.Label(
            master,
            text="EMA Length:",
        )

        self.macd_params_label = tk.Label(
            master,
            text="MACD",
            bg="#333333",
            fg="#FFFFFF",
        )

        self.label_macd_source = tk.Label(
            master,
            text="MACD Source Column:",
        )

        self.label_macd_fast_length = tk.Label(
            master,
            text="MACD Fast Length:",
        )

        self.label_macd_slow_length = tk.Label(
            master,
            text="MACD Slow Length:",
        )

        self.label_macd_signal_length = tk.Label(
            master,
            text="MACD Signal Length:",
        )

        self.cci_params_label = tk.Label(
            master,
            text="CCI",
            bg="#333333",
            fg="#FFFFFF",
        )

        self.label_cci_source = tk.Label(
            master,
            text="CCI Source Column:",
        )

        self.label_cci_length = tk.Label(
            master,
            text="CCI Length:",
        )

        self.label_cci_ma_type = tk.Label(
            master,
            text="CCI MA Type:",
        )

        self.label_cci_constant = tk.Label(
            master,
            text="CCI Constant:",
        )

        self.irb_params_label = tk.Label(
            master,
            text="IRB Params",
            bg="#333333",
            fg="#FFFFFF",
        )

        self.irb_lowestlow_label = tk.Label(
            master,
            text="Lowest Low:",
        )

        self.irb_payoff_label = tk.Label(
            master,
            text="Payoff:",
        )

        self.irb_tick_size_label = tk.Label(
            master,
            text="Tick Size:",
        )

        self.irb_wick_percentage_label = tk.Label(
            master,
            text="Wick Percentage:",
        )

        self.trend_params_label = tk.Label(
            master,
            text="Trend Params",
            bg="#333333",
            fg="#FFFFFF",
        )

        self.trend_ema_label = tk.Label(
            master,
            text="EMA:",
        )

        self.trend_cci_label = tk.Label(
            master,
            text="CCI:",
        )

        self.trend_macd_label = tk.Label(
            master,
            text="MACD:",
        )

        self.indicators_params_label = tk.Label(
            master,
            text="Indicators Params",
            bg="#333333",
            fg="#FFFFFF",
        )

        self.indicators_ema_column_label = tk.Label(
            master,
            text="EMA Column:",
        )

        self.indicators_macd_histogram_trend_value_label = tk.Label(
            master,
            text="MACD Histogram Trend Value:",
        )

        self.indicators_cci_trend_value_label = tk.Label(
            master,
            text="CCI Trend Value:",
        )


class Grid(Label):
    def __init__(self, master):
        Label.__init__(self, master)

        self.label1.grid(
            row=0,
            column=0,
        )

        self.label2.grid(
            row=0,
            column=2,
        )

        self.ema_params_label.grid(
            row=4,
            column=0,
            columnspan=4,
            sticky="NSWE",
        )

        self.label_ema_source.grid(
            row=5,
            column=0,
        )

        self.label_ema_length.grid(
            row=5,
            column=2,
        )

        self.macd_params_label.grid(
            row=6,
            column=0,
            columnspan=4,
            sticky="NSWE",
        )

        self.label_macd_source.grid(
            row=7,
            column=0,
        )

        self.label_macd_fast_length.grid(
            row=7,
            column=2,
        )

        self.label_macd_slow_length.grid(
            row=8,
            column=0,
        )

        self.label_macd_signal_length.grid(
            row=8,
            column=2,
        )

        self.cci_params_label.grid(
            row=9,
            column=0,
            columnspan=4,
            sticky="NSWE",
        )

        self.label_cci_source.grid(
            row=10,
            column=0,
        )

        self.label_cci_length.grid(
            row=10,
            column=2,
        )

        self.label_cci_ma_type.grid(
            row=11,
            column=0,
        )

        self.label_cci_constant.grid(
            row=11,
            column=2,
        )

        self.irb_params_label.grid(
            row=12,
            column=0,
            columnspan=4,
            sticky="NSWE",
        )

        self.irb_lowestlow_label.grid(
            row=13,
            column=0,
        )

        self.irb_payoff_label.grid(
            row=13,
            column=2,
        )

        self.irb_tick_size_label.grid(
            row=14,
            column=0,
        )

        self.irb_wick_percentage_label.grid(
            row=14,
            column=2,
        )

        self.trend_params_label.grid(
            row=15,
            column=0,
            columnspan=4,
            sticky="NSWE",
        )

        self.trend_ema_label.grid(
            row=16,
            column=0,
        )

        self.trend_cci_label.grid(
            row=16,
            column=2,
        )

        self.trend_macd_label.grid(
            row=17,
            column=0,
        )

        self.indicators_params_label.grid(
            row=18,
            column=0,
            columnspan=4,
            sticky="NSWE",
        )

        self.indicators_ema_column_label.grid(
            row=19,
            column=0,
        )

        self.indicators_macd_histogram_trend_value_label.grid(
            row=19,
            column=2,
        )

        self.indicators_cci_trend_value_label.grid(
            row=20,
            column=0,
        )
