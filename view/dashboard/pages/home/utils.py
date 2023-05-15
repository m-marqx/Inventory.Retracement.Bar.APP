import dash_bootstrap_components as dbc

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

indicators_filter = [
    {"label": "EMA", "value": "ema"},
    {"label": "CCI", "value": "cci"},
    {"label": "MACD", "value": "macd"},
]
