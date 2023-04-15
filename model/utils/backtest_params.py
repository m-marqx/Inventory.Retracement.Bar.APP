import numpy as np
from pydantic import BaseModel
from typing import List


class EmaParamsBacktest(BaseModel):
    length: List[int] = np.arange(1, 100 + 1, 1)
    source_column: List[str] = ["open", "high", "low", "close"]


class IrbParamsBacktest(BaseModel):
    lowestlow: List[int] = np.arange(1, 2, 1)
    payoff: List[int] = np.arange(2, 3, 1)
    ticksize: List[float] = np.arange(0.1, 0.2)
    wick_percentage: List[float] = np.round(np.arange(0.01, 1.01, 0.01), 2)


class IndicatorsParamsBacktest(BaseModel):
    ema_column: List[str] = ["open", "high", "low", "close"]
    macd_histogram_trend_value: List[int] = np.arange(0, 1, 1)
    cci_trend_value: List[int] = np.arange(0, 1, 1)


class TrendParamsBacktest(BaseModel):
    ema: List[bool] = [True, False]
    macd: List[bool] = [True, False]
    cci: List[bool] = [True, False]


