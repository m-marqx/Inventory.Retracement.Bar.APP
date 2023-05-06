from controller.api.coin_futures import CoinMargined
import dash_bootstrap_components as dbc

from model.strategy.params.indicators_params import (
    EmaParams,
    MACDParams,
    CCIParams,
)
from model.strategy.params.strategy_params import (
    TrendParams,
    IrbParams,
    IndicatorsParams,
)

from model.strategy.indicators import BuilderSource
from model.strategy.strategy import BuilderStrategy
from pydantic import BaseModel


ema_ohlc_items = [
    dbc.DropdownMenuItem("Close", id="ema_close"),
    dbc.DropdownMenuItem("Open", id="ema_open"),
    dbc.DropdownMenuItem("High", id="ema_high"),
    dbc.DropdownMenuItem("Low", id="ema_low"),
]

macd_ohlc_items = [
    dbc.DropdownMenuItem("Close", id="macd_close"),
    dbc.DropdownMenuItem("Open", id="macd_open"),
    dbc.DropdownMenuItem("High", id="macd_high"),
    dbc.DropdownMenuItem("Low", id="macd_low"),
]

cci_ohlc_items = [
    dbc.DropdownMenuItem("Close", id="cci_close"),
    dbc.DropdownMenuItem("Open", id="cci_open"),
    dbc.DropdownMenuItem("High", id="cci_high"),
    dbc.DropdownMenuItem("Low", id="cci_low"),
]

source_ohlc_items = [
    dbc.DropdownMenuItem("Close", id="source_close"),
    dbc.DropdownMenuItem("Open", id="source_open"),
    dbc.DropdownMenuItem("High", id="source_high"),
    dbc.DropdownMenuItem("Low", id="source_low"),
]

cci_ma_type_items = [
    dbc.DropdownMenuItem("SMA", id="sma"),
    dbc.DropdownMenuItem("EMA", id="ema"),
]

intervals = [
    dbc.DropdownMenuItem("1min", id="1m"),
    dbc.DropdownMenuItem("5min", id="5m"),
    dbc.DropdownMenuItem("15min", id="15m"),
    dbc.DropdownMenuItem("30min", id="30m"),
    dbc.DropdownMenuItem("1h", id="1h"),
    dbc.DropdownMenuItem("2h", id="2h"),
    dbc.DropdownMenuItem("4h", id="4h"),
    dbc.DropdownMenuItem("6h", id="6h"),
    dbc.DropdownMenuItem("8h", id="8h"),
    dbc.DropdownMenuItem("12h", id="12h"),
    dbc.DropdownMenuItem("1d", id="1d"),
    dbc.DropdownMenuItem("3d", id="3d"),
    dbc.DropdownMenuItem("1w", id="1w"),
    dbc.DropdownMenuItem("1M", id="1M"),
]

indicators_filter = [
    {"label": "EMA", "value": "ema"},
    {"label": "CCI", "value": "cci"},
    {"label": "MACD", "value": "macd"},
]


def get_data(symbol, interval):
    symbol = symbol.upper()  # Avoid errors when the symbol is in lowercase
    if symbol.endswith("USD"):
        symbol += "_PERP"

    fapi = CoinMargined(symbol, interval)
    data_frame = fapi.get_Klines().get_all_futures_klines_df()
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
