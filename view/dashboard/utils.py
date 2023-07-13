from pydantic import BaseModel
import pandas as pd

from controller.api.klines_api import KlineAPI

from model.strategy.params import (
    EmaParams,
    MACDParams,
    CCIParams,
    TrendParams,
    IrbParams,
    IndicatorsParams,
    ResultParams,
)

from model.strategy.indicators import BuilderSource
from model.strategy.strategy import BuilderStrategy


def get_data(symbol: str, interval: str, api: str) -> pd.DataFrame:
    """Fetches historical OHLC data for a symbol and interval.

    Parameters
    ----------
    symbol : str
        The symbol of the asset for which the data is to be fetched.
    interval : str
        The time interval of the OHLC data (e.g., '1h', '4h', '1d').
    api : API
        An instance of the API class used for fetching data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the OHLC data.

    """
    fapi = KlineAPI(symbol, interval, api)
    data_frame = fapi.get_Klines().to_OHLC_DataFrame()
    return data_frame


class BuilderParams(BaseModel):
    """A class representing builder parameters for a trading strategy.

    Attributes
    ----------
    ema_params : EmaParams
        The parameters for the Exponential Moving Average (EMA)
        indicator.
    macd_params : MACDParams
        The parameters for the Moving Average Convergence Divergence
        (MACD) indicator.
    cci_params : CCIParams
        The parameters for the Commodity Channel Index (CCI) indicator.
    irb_params : IrbParams
        The parameters for the Inside Bar Reversal (IRB) indicator.
    indicator_params : IndicatorsParams
        The parameters for general indicators.
    trend_params : TrendParams
        The parameters for trend indicators.
    result_params : ResultParams
        The parameters for the strategy results.

    """
    ema_params = EmaParams()
    macd_params = MACDParams()
    cci_params = CCIParams()
    irb_params = IrbParams()
    indicator_params = IndicatorsParams()
    trend_params = TrendParams(ema=True, macd=True, cci=True)
    result_params = ResultParams()


def builder(
    data_frame: pd.DataFrame,
    params: BuilderParams = BuilderParams(),
) -> pd.DataFrame:
    """Builds a strategy based on the input data frame and parameters.

    Parameters
    ----------
    data_frame : pd.DataFrame
        The input data frame containing the data for building the
        strategy.

    params : BuilderParams, optional
        The parameters for building the strategy. If not provided,
        default parameters defined in `BuilderParams` will be used.

    Returns
    -------
    pd.DataFrame
        The resulting data frame containing the built strategy.

    """
    df_source = BuilderSource(data_frame)
    if params.trend_params.ema:
        df_source = df_source.set_EMA_params(params.ema_params).set_ema()
    if params.trend_params.macd:
        df_source = df_source.set_MACD_params(params.macd_params).set_macd()
    if params.trend_params.cci:
        df_source = df_source.set_CCI_params(params.cci_params).set_cci()
    df_source = df_source.execute()

    df_strategy = (
        BuilderStrategy(df_source)
        .set_trend_params(params.indicator_params, params.trend_params)
        .get_trend()
        .set_irb_params(params.irb_params)
        .get_irb_signals()
        .calculate_irb_signals()
        .set_result_params(params.result_params)
        .calculateResults()
        .execute()
    )

    return df_strategy
