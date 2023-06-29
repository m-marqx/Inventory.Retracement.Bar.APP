from pydantic import BaseModel


class EmaParams(BaseModel):
    """
    Parameters for the EMA strategy.

    Parameters
    ----------
    source_column : str, optional
        The source column to use for calculations (default is "close").
    length : int, optional
        The length of the EMA window (default is 20).
    """

    source_column: str = "close"
    length: int = 20


class MACDParams(BaseModel):
    """
    Parameters for the MACD strategy.

    Parameters
    ----------
    source_column : str, optional
        The source column to use for calculations (default is "close").
    fast_length : int, optional
        The length of the fast EMA window (default is 35).
    slow_length : int, optional
        The length of the slow EMA window (default is 100).
    signal_length : int, optional
        The length of the signal EMA window (default is 8).
    """

    source_column: str = "close"
    fast_length: int = 35
    slow_length: int = 100
    signal_length: int = 8


class CCIParams(BaseModel):
    """
    Parameters for the CCI strategy.

    Parameters
    ----------
    source_column : str, optional
        The source column to use for calculations (default is "close").
    length : int, optional
        The length of the CCI window (default is 20).
    ma_type : str, optional
        The type of moving average to use (default is "sma").
    constant : float, optional
        The constant value used in CCI calculation (default is 0.015).
    """

    source_column: str = "close"
    length: int = 20
    ma_type: str = "sma"
    constant: float = 0.015
