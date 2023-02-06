import pandas as pd
from moving_average import moving_average

ma = moving_average()


class indicators:
    def CCI(self, source, length):
        df = pd.DataFrame()
        df["TP"] = source
        df["sma"] = df["TP"].rolling(length).mean()
        df["mad"] = df["TP"].rolling(length).apply(lambda x: pd.Series(x).mad())
        df["CCI"] = (df["TP"] - df["sma"]) / (0.015 * df["mad"])
        return df["CCI"].dropna(axis=0)

    def MACD(self, source, fast_length, slow_length, signal_length):
        MACD = MA.ema(source, fast_length) - MA.ema(source, slow_length)
        df = pd.DataFrame()
        df["MACD"] = pd.DataFrame(MACD).dropna(axis=0)
        macd_Signal = pd.DataFrame()
        macd_Signal["MACD"] = df["MACD"]
        macd_Signal["MACD_Signal"] = MA.ema(macd_Signal["MACD"], signal_length)
        macd_Signal["Histogram"] = macd_Signal["MACD"] - macd_Signal["MACD_Signal"]
        return macd_Signal["Histogram"]
