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

    def strategy(
        self,
        ema_params: EmaParams,
        irb_params: IrbParams,
        indicators: IndicatorsParams,
        trend: TrendParams,
    ):
        self.data_frame = (
            BuilderSource(
                self.dataframe,
            )
            .set_EMA_params(ema_params)
            .set_ema()
            .execute()
        )
        self.data_frame = (
            BuilderStrategy(
                self.data_frame,
            )
            .set_trend_params(indicators, trend)
            .get_trend()
            .set_irb_params(irb_params)
            .get_irb_signals()
            .calculate_irb_signals()
            .calculateResults_new()
            .execute()
        )
