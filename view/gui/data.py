import tkinter as tk
import pandas as pd
from controller.future_API import FuturesAPI
from .utils import Text


class GetDataGUI(Text):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.df = pd.DataFrame()

        self.symbol_entry = tk.Entry(self.master, width=10)
        self.symbol_entry.grid(row=0, column=1)
        self.timeframes = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ]

        self.timeframe_var = tk.StringVar(self.master)
        self.timeframe_var.set(self.timeframes[6])

        self.timeframe_menu = tk.OptionMenu(
            self.master, self.timeframe_var, *self.timeframes
        )
        self.timeframe_menu.grid(row=0, column=3)

        self.get_data_button = tk.Button(
            self.master, text="Get Data", command=self.get_data
        )
        self.get_data_button.grid(row=1, column=1)

        self.master = master

    def get_data(self):
        symbol = self.symbol_entry.get()
        interval = self.timeframe_var.get()

        try:
            # Change button text to "loading"
            self.get_data_button.config(state="disabled", text="Loading...")
            self.master.update()

            fapi = FuturesAPI()
            self.df = fapi.get_all_futures_klines_df(symbol, interval)

            self.insert_text(
                f"Retrieved {len(self.df)} klines for {symbol} "
                f"with interval {interval}\n"
            )

        except Exception as exception:
            self.insert_text(f"Error occurred: {str(exception)}\n")

        finally:
            # Change button text back to "Get Data"
            self.get_data_button.config(state="normal", text="Get Data")
            self.master.update()

        return self
