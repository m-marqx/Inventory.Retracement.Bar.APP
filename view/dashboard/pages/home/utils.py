import dash_bootstrap_components as dbc


class DropdownMenuItems:
    def __init__(self, lang):
        self.lang = lang

    @property
    def ema_ohlc_items(self):
        return [
            dbc.DropdownMenuItem(self.lang["OPEN"], id="ema_open"),
            dbc.DropdownMenuItem(self.lang["HIGH"], id="ema_high"),
            dbc.DropdownMenuItem(self.lang["LOW"], id="ema_low"),
            dbc.DropdownMenuItem(self.lang["CLOSE"], id="ema_close"),
        ]

    @property
    def macd_ohlc_items(self):
        return [
            dbc.DropdownMenuItem(self.lang["OPEN"], id="macd_open"),
            dbc.DropdownMenuItem(self.lang["HIGH"], id="macd_high"),
            dbc.DropdownMenuItem(self.lang["LOW"], id="macd_low"),
            dbc.DropdownMenuItem(self.lang["CLOSE"], id="macd_close"),
        ]

    @property
    def cci_ohlc_items(self):
        return [
                dbc.DropdownMenuItem(self.lang["OPEN"], id="cci_open"),
                dbc.DropdownMenuItem(self.lang["HIGH"], id="cci_high"),
                dbc.DropdownMenuItem(self.lang["LOW"], id="cci_low"),
                dbc.DropdownMenuItem(self.lang["CLOSE"], id="cci_close"),
        ]

    @property
    def source_ohlc_items(self):
        return [
            dbc.DropdownMenuItem(self.lang["OPEN"], id="source_open"),
            dbc.DropdownMenuItem(self.lang["HIGH"], id="source_high"),
            dbc.DropdownMenuItem(self.lang["LOW"], id="source_low"),
            dbc.DropdownMenuItem(self.lang["CLOSE"], id="source_close"),
        ]

    @property
    def cci_ma_type_items(self):
        return [
            dbc.DropdownMenuItem("SMA", id="sma"),
            dbc.DropdownMenuItem("EMA", id="ema"),
        ]

    @property
    def indicators_filter(self):
        return [
            {"label": "EMA", "value": "ema"},
            {"label": "CCI", "value": "cci"},
            {"label": "MACD", "value": "macd"},
        ]
