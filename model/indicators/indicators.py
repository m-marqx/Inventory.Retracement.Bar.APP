import pandas as pd
import numpy as np
from model import MovingAverage

ma = MovingAverage()


class MACD:
    def __init__(self, source, fast_length, slow_length, signal_length):
        self.source = source
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length

    def set_ema(self):
        self.fast_ma = ma.ema(self.source, self.fast_length)
        self.slow_ma = ma.ema(self.source, self.slow_length)

        return self

    def set_sma(self):
        self.fast_ma = ma.sma(self.source, self.fast_length)
        self.slow_ma = ma.sma(self.source, self.slow_length)

        return self

    def MACD(self):
        self.MACD = self.fast_ma - self.slow_ma
        self.df = pd.DataFrame({"MACD": self.MACD}).dropna(axis=0)
        self.df["MACD_Signal"] = ma.ema(self.df["MACD"], self.signal_length)
        self.df["Histogram"] = self.df["MACD"] - self.df["MACD_Signal"]

        return self.df


class CCI:
    def __init__(self, source, length: int = 20):
        self.source_arr = np.array(source)
        self.source_df = pd.DataFrame({"source_arr": source})
        self.length = length

    def CCI_precise(
    #! this version have more similar results from excel 
    #! than the other version and TA-lib.
        self, 
        smooth_column: str = "sma", 
        constant: float = 0.015,
    ):
        self.df = pd.DataFrame()
        self.df["TP"] = self.source_arr
        self.df["sma"] = self.df["TP"].rolling(self.length).mean()

        self.df["mad"] = (
            self.df["TP"]
            .rolling(self.length)
            .apply(lambda x: (pd.Series(x) - pd.Series(x).mean()).abs().mean())
        )

        self.df["CCI"] = (
            (self.df["TP"] - self.df[smooth_column]) 
            / (constant * self.df["mad"])
        )

        self.df["CCI"].dropna(axis=0, inplace=True)

        return self