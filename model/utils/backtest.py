import pandas as pd
from model.strategy.params.indicators_params import (
    EmaParams,
    MACDParams,
    CCIParams,
)

from model.strategy.params.strategy_params import (
    IrbParams,
    IndicatorsParams,
)

from model.strategy.strategy import BuilderStrategy
from model.strategy.indicators import BuilderSource

class Backtest:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()
        self.data_frame = pd.DataFrame()
