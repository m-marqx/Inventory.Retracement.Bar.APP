import dash_bootstrap_components as dbc

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

def api_types(lang):
    return [
        {"label": lang["SPOT"], "value": "spot"},
        {"label": lang["FUTURES"], "value": "coin_margined"},
        {"label": lang["MARK_PRICE"], "value": "mark_price"},
    ]

def result_types(lang):
    return [
        {"label": lang["FIXED"], "value": "fixed"},
        {"label": lang["NORMAL"], "value": "normal"},
    ]
