from pydantic import BaseModel


class EmaParams(BaseModel):
    source_column: str = "close"
    length: int = 20


class MACDParams(BaseModel):
    source_column: str = "close"
    fast_length: int = 35
    slow_length: int = 100
    signal_length: int = 8


class CCIParams(BaseModel):
    source_column: str = "close"
    length: int = 20
    ma_type: str = "sma"
    constant: int = 0.015
