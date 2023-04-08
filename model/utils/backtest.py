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

    def ema_backtest(self, start=0, end=100, column="Cumulative_Result"):
        df_result = {}
        columns = ["open", "high", "low", "close"]
        for col in columns:
            for value in range(start, end + 1, 1):
                ema_params = EmaParams(length=value, source_column=col)
                arr = self.strategy(self.data_frame, ema_params, IrbParams())[column]
                df_result[f"length: {value} <br> source: {col}"] = arr

        df_result = pd.DataFrame(df_result)
        return df_result

