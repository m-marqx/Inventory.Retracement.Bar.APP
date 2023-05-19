import numpy as np
from pydantic import BaseModel
from typing import List


class EmaParamsBacktest(BaseModel):
    length: List[int] = range(1, 100 + 1, 1)
    source_column: List[str] = ["open", "high", "low", "close"]


class IrbParamsBacktest(BaseModel):
    lowestlow: List[int] = range(1, 2, 1)
    payoff: List[int] = range(2, 3, 1)
    ticksize: List[float] = np.arange(0.1, 0.2)
    wick_percentage: List[float] = np.round(np.arange(0.01, 1.01, 0.01), 2)


class IndicatorsParamsBacktest(BaseModel):
    ema_column: List[str] = ["open", "high", "low", "close"]
    macd_histogram_trend_value: List[int] = range(0, 1, 1)
    cci_trend_value: List[int] = range(0, 1, 1)


class TrendParamsBacktest(BaseModel):
    ema: List[bool] = [True]
    macd: List[bool] = [False]
    cci: List[bool] = [False]


class BacktestParams(BaseModel):
    ema_params: EmaParamsBacktest = EmaParamsBacktest()
    irb_params: IrbParamsBacktest = IrbParamsBacktest()
    indicators_params: IndicatorsParamsBacktest = IndicatorsParamsBacktest()
    trend_params: TrendParamsBacktest = TrendParamsBacktest()

    @property
    def total_combinations(self):
        num_combinations = (
            len(self.ema_params.length)
            * len(self.ema_params.source_column)
            * len(self.irb_params.lowestlow)
            * len(self.irb_params.payoff)
            * len(self.irb_params.ticksize)
            * len(self.irb_params.wick_percentage)
            * len(self.indicators_params.ema_column)
            * len(self.indicators_params.macd_histogram_trend_value)
            * len(self.indicators_params.cci_trend_value)
            * len(self.trend_params.ema)
            * len(self.trend_params.macd)
            * len(self.trend_params.cci)
        )
        return num_combinations
