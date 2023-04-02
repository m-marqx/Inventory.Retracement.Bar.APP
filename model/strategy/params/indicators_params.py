from pydantic import BaseModel

class EMA_params(BaseModel):
        source_column: str = 'close'
        length: int = 20

class MACD_params(BaseModel):
    source_column: str = "close"
    fast_length: int = 35
    slow_length: int = 100
    signal_length: int = 8

class CCI_params(BaseModel):
    source_column: str = "close"
    length: int = 20
    ma_type: str = "sma"