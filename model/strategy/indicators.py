from pydantic import BaseModel
from model.utils import clean_data
from model.strategy.strategy import BaseStrategy
from model.strategy.params.indicators_params import EMA_params, MACD_params, CCI_params


class calculate_EMA(BaseStrategy):
    def __init__(self, dataframe, params: EMA_params):
        super().__init__(dataframe)
        self.source = self.df_filtered[params.source_column]
        self.length = params.length

    def execute(self):
        from model.indicators.moving_average import MovingAverage
        ma = MovingAverage()
        self.ema = ma.ema(self.source, self.length)
        self.df_filtered['ema'] = self.ema
        return self.df_filtered

class calculate_MACD(BaseStrategy):
    def __init__(self, dataframe,  params: MACD_params):
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

class calculate_CCI(BaseStrategy): 
    def __init__(self, dataframe, params: CCI_params):
        super().__init__(dataframe)
        self.source = self.df_filtered[params.source_column]
        self.length = params.length
        self.ma_type = params.ma_type

    def execute(self):
        from model.indicators.CCI import CCI
        self.CCI = CCI(self.source, self.length)

        if self.ma_type == "sma":
            self.CCI.set_sma()
        if self.ma_type == "ema":
            self.CCI.set_ema()

        self.df_filtered['CCI'] = self.CCI.CCI()['CCI']
        self.df_filtered['CCI'].shift(self.length - 1)
        return self.df_filtered

class builder_source(BaseStrategy):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.df_filtered = clean_data(self.df_filtered).execute()

    def set_EMA_params(self, params: EMA_params):
        self.ema_params = params
        return self

    def set_ema(self):
        calculate_EMA(self.df_filtered,self.ema_params).execute()
        return self

    def set_CCI_params(self, params: CCI_params):
        self.cci_params = params
        return self

    def set_cci(self):
        calculate_CCI(self.df_filtered,self.cci_params).execute()
        return self

    def set_MACD_params(self, params: MACD_params):
        self.macd_params = params
        return self

    def set_macd(self):
        calculate_MACD(self.df_filtered, self.macd_params).execute()
        return self

    def execute(self):
        return self.df_filtered