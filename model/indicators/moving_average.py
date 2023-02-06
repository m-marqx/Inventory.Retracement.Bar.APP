import pandas as pd

# Indicadores
class moving_average:
    def sma(self, source, length):
        sma = source.rolling(length).mean()
        return sma.dropna(axis=0)

    def ema(self, source, length):
        sma = source.rolling(window=length, min_periods=length).mean()[:length]
        rest = source[length:]
        return (
            pd.concat([sma, rest]).ewm(span=length, adjust=False).mean().dropna(axis=0)
        )
