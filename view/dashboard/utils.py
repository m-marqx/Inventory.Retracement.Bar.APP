from controller.api.klines_api import KlineAPI

from model.strategy.params import (
    EmaParams,
    MACDParams,
    CCIParams,
    TrendParams,
    IrbParams,
    IndicatorsParams,
)

from model.strategy.indicators import BuilderSource
from model.strategy.strategy import BuilderStrategy
from pydantic import BaseModel


def get_data(symbol, interval, api):
    fapi = KlineAPI(symbol, interval, api)
    data_frame = fapi.get_Klines().to_OHLC_DataFrame()
    return data_frame


class BuilderParams(BaseModel):
    ema_params = EmaParams()
    macd_params = MACDParams()
    cci_params = CCIParams()
    irb_params = IrbParams()
    indicator_params = IndicatorsParams()
    trend_params = TrendParams(ema=True, macd=True, cci=True)


def builder(data_frame, params: BuilderParams()):
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
        .calculateResults()
        .execute()
    )

    return df_strategy
