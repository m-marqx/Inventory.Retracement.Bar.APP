from model.utils import CleanData, BaseStrategy
from model.strategy.params.indicators_params import (
    EmaParams,
    MACDParams,
    CCIParams
)


class CalculateEma(BaseStrategy):
    def __init__(self, dataframe, params: EmaParams):
        super().__init__(dataframe)
        self.source = self.df_filtered[params.source_column]
        self.length = params.length

    def execute(self):
        from model.indicators.moving_average import MovingAverage
        ma = MovingAverage()
        self.ema = ma.Ema(self.source, self.length)
        self.df_filtered["ema"] = self.ema
        return self.df_filtered

class CalculateMACD(BaseStrategy):
    def __init__(self, dataframe, params: MACDParams):
        super().__init__(dataframe)
        self.source = self.df_filtered[params.source_column]
        self.fast_length = params.fast_length
        self.slow_length = params.slow_length
        self.signal_length = params.signal_length

    def execute(self):
        from model.indicators.MACD import MACD
        self.histogram = MACD(self.source, self.fast_length, self.slow_length, self.signal_length).set_ema().MACD()['Histogram']
        self.df_filtered['MACD_Histogram'] = self.histogram
        return self.df_filtered

class CalculateCCI(BaseStrategy):
    def __init__(self, dataframe, params: CCIParams):
        super().__init__(dataframe)
        self.source = self.df_filtered[params.source_column]
        self.params = params

    def execute(self):
        from model.indicators.CCI import CCI
        self.CCI = CCI(self.source, self.params.length)

        if self.params.ma_type == "sma":
            self.CCI.set_sma()
        if self.params.ma_type == "ema":
            self.CCI.set_ema()

        self.df_filtered["CCI"] = self.CCI.CCI(self.params.constant)["CCI"]
        return self.df_filtered

class BuilderSource(BaseStrategy):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.df_filtered = CleanData(self.df_filtered).execute()

    def set_EMA_params(self, params: EmaParams = EmaParams()):
        self.ema_params = params
        return self

    def set_ema(self):
        CalculateEma(self.df_filtered, self.ema_params).execute()
        return self

    def set_CCI_params(self, params: CCIParams = CCIParams()):
        self.cci_params = params
        return self

    def set_cci(self):
        CalculateCCI(self.df_filtered, self.cci_params).execute()
        return self

    def set_MACD_params(self, params: MACDParams = MACDParams()):
        self.macd_params = params
        return self

    def set_macd(self):
        CalculateMACD(self.df_filtered, self.macd_params).execute()
        return self

    def execute(self):
        return self.df_filtered
