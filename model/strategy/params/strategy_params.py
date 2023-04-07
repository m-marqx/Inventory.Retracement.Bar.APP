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
