from pydantic import BaseModel

class indicators_params(BaseModel):
    ema_column: str = 'close'
    macd_histogram_trend_value: int = 0
    cci_trend_value: int = 0 

class trend_params(BaseModel):
    ema: bool = False
    cci: bool = False
    macd: bool = False
    trend: bool = False

class irb_params(BaseModel):
    lowestlow: float = 1
    payoff: float = 2
    tick_size: float = 0.1
    wick_percentage: float = 0.45
