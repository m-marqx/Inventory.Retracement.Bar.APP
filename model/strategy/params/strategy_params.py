from pydantic import BaseModel


class IndicatorsParams(BaseModel):
    """
    Parameters for indicators used in trading strategies.

    Parameters
    ----------
    ema_column : str, optional
        The column to use for EMA calculations (default is "close").
    macd_histogram_trend_value : float, optional
        The threshold value for MACD histogram trend (default is 0).
    cci_trend_value : float, optional
        The threshold value for CCI trend (default is 0).
    """

    ema_column: str = "close"
    macd_histogram_trend_value: float = 0
    cci_trend_value: float = 0


class TrendParams(BaseModel):
    """
    Parameters for defining trading trends.

    Parameters
    ----------
    ema : bool, optional
        Flag indicating the usage of EMA (default is False).
    cci : bool, optional
        Flag indicating the usage of CCI (default is False).
    macd : bool, optional
        Flag indicating the usage of MACD (default is False).
    trend : bool, optional
        Flag indicating the usage of trend analysis (default is False).
    """

    ema: bool = False
    cci: bool = False
    macd: bool = False
    trend: bool = False


class IrbParams(BaseModel):
    """
    Parameters for the Inventory Retracement Bar strategy.

    Parameters
    ----------
    lowestlow : int, optional
        The lowest low value (default is 1).
    payoff : float, optional
        The payoff value (default is 2).
    tick_size : float, optional
        The tick size value (default is 0.1).
    wick_percentage : float, optional
        The wick percentage value (default is 0.45).
    """

    lowestlow: int = 1
    payoff: float = 2
    tick_size: float = 0.1
    wick_percentage: float = 0.45


class ResultParams(BaseModel):
    """
    Parameters for result calculations.

    Parameters
    ----------
    capital : float, optional
        The initial capital amount (default is 100000).
    percent : bool, optional
        Flag indicating the usage of percent values (default is True).
    gain : float, optional
        The gain value (default is 2).
    loss : float, optional
        The loss value (default is -1).
    method : str, optional
        The method for result calculation (default is "Fixed").
    qty : float, optional
        The quantity value (default is 1).
    coin_margined : bool, optional
        Flag indicating coin margined trading (default is False).
    """

    capital: float = 100_000
    percent: bool = True
    gain: float = 2
    loss: float = -1
    method: str = "Fixed"
    qty: float = 1
    coin_margined: bool = False
