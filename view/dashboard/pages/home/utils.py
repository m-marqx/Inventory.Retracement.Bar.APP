import dash_bootstrap_components as dbc
from view.dashboard.pages.lang.en_US import lang


ema_ohlc_items = [
    dbc.DropdownMenuItem(lang["OPEN"], id="ema_open"),
    dbc.DropdownMenuItem(lang["HIGH"], id="ema_high"),
    dbc.DropdownMenuItem(lang["LOW"], id="ema_low"),
    dbc.DropdownMenuItem(lang["CLOSE"], id="ema_close"),
]

macd_ohlc_items = [
    dbc.DropdownMenuItem(lang["OPEN"], id="macd_open"),
    dbc.DropdownMenuItem(lang["HIGH"], id="macd_high"),
    dbc.DropdownMenuItem(lang["LOW"], id="macd_low"),
    dbc.DropdownMenuItem(lang["CLOSE"], id="macd_close"),
]

cci_ohlc_items = [
    dbc.DropdownMenuItem(lang["OPEN"], id="cci_open"),
    dbc.DropdownMenuItem(lang["HIGH"], id="cci_high"),
    dbc.DropdownMenuItem(lang["LOW"], id="cci_low"),
    dbc.DropdownMenuItem(lang["CLOSE"], id="cci_close"),
]

source_ohlc_items = [
    dbc.DropdownMenuItem(lang["OPEN"], id="source_open"),
    dbc.DropdownMenuItem(lang["HIGH"], id="source_high"),
    dbc.DropdownMenuItem(lang["LOW"], id="source_low"),
    dbc.DropdownMenuItem(lang["CLOSE"], id="source_close"),
]

cci_ma_type_items = [
    dbc.DropdownMenuItem("SMA", id="sma"),
    dbc.DropdownMenuItem("EMA", id="ema"),
]

indicators_filter = [
    {"label": "EMA", "value": "ema"},
    {"label": "CCI", "value": "cci"},
    {"label": "MACD", "value": "macd"},
]
