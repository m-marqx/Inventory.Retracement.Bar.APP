from pydantic import BaseModel

class IndicatorsParams(BaseModel):
    ema_column: str = "close"
    macd_histogram_trend_value: float = 0
    cci_trend_value: float = 0

class TrendParams(BaseModel):
    ema: bool = False
    cci: bool = False
    macd: bool = False
    trend: bool = False

class IrbParams(BaseModel):
    lowestlow: int = 1
    payoff: float = 2
    tick_size: float = 0.1
    wick_percentage: float = 0.45

class ResultParams(BaseModel):
    capital: float = 100_000
    percent: bool = True
    gain: float = 2
    loss: float = -1
    method: str = "fixed"
    qty: float = 1
    coin_margined: bool = False
