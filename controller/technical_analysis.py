# Import libraries
import pandas as pd

# Indicadores
class indicators:
    def sma(self, source, length):
        sma = source.rolling(length).mean()
        return sma.dropna(axis=0)

    def ema(self, source, periods):
        sma = source.rolling(window=periods, min_periods=periods).mean()[:periods]
        rest = source[periods:]
        return (
            pd.concat([sma, rest]).ewm(span=periods, adjust=False).mean().dropna(axis=0)
        )

    def CCI(self, source, length):
        df = pd.DataFrame()
        df["TP"] = source
        df["sma"] = df["TP"].rolling(length).mean()
        df["mad"] = df["TP"].rolling(length).apply(lambda x: pd.Series(x).mad())
        df["CCI"] = (df["TP"] - df["sma"]) / (0.015 * df["mad"])
        return df["CCI"].dropna(axis=0)

    def MACD(self, source, fast_length, slow_length, signal_length):
        MACD = self.ema(source, fast_length) - self.ema(source, slow_length)
        df = pd.DataFrame()
        df["MACD"] = pd.DataFrame(MACD).dropna(axis=0)
        macd_Signal = pd.DataFrame()
        macd_Signal["MACD"] = df["MACD"]
        macd_Signal["MACD_Signal"] = self.ema(macd_Signal["MACD"], signal_length)
        macd_Signal["Histogram"] = macd_Signal["MACD"] - macd_Signal["MACD_Signal"]
        return macd_Signal["Histogram"]
