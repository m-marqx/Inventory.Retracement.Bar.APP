
import pandas as pd
from model.indicators import MovingAverage

ma = MovingAverage()


def RSI(source: pd.Series, periods: int=14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given time series data.

    Parameters:
    -----------
    source : pd.Series
        The input time series data for which to calculate RSI.
    periods : int, optional
        The number of periods to use for RSI calculation, by default 14.

    Returns:
    --------
    pd.Series
        The calculated RSI values for the input data.
    """
    change = source.diff()
    upward_diff = max(change - change.shift(-1), change, 0.0)
    downward_diff = max(change.shift(-1) - change, change, 0.0)

    relative_strength = (
        ma.rma(upward_diff, periods)
        / ma.rma(downward_diff, periods)
    )

    rsi = 100 - (100 / (1 + relative_strength))
    return rsi
