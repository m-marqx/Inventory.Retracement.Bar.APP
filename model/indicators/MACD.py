import pandas as pd
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
        self.data_frame = pd.DataFrame({"MACD": self.MACD}).dropna(axis=0)
        self.data_frame["MACD_Signal"] = ma.ema(self.data_frame["MACD"], self.signal_length)
        self.data_frame["Histogram"] = self.data_frame["MACD"] - self.data_frame["MACD_Signal"]

        return self.data_frame
