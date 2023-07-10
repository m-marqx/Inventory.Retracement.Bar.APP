import dash
from dash import Output, Input, State, callback

class LabelCallbacks:
    @callback(
        Output("ema_source_column", "label"),
        Input("ema_open", "n_clicks"),
        Input("ema_high", "n_clicks"),
        Input("ema_low", "n_clicks"),
        Input("ema_close", "n_clicks"),
        State("ema_source_column", "label"),
    )
    def update_ema_label(open_clicks, high_clicks, low_clicks, close_clicks, ema_source):
        ctx = dash.callback_context
        if not ctx.triggered:
            return ema_source
        else:
            ema_button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if ema_button_id == "ema_close":
                return "close"
            elif ema_button_id == "ema_open":
                return "open"
            elif ema_button_id == "ema_high":
                return "high"
            elif ema_button_id == "ema_low":
                return "low"


    @callback(
        Output("macd_source_column", "label"),
        Input("macd_open", "n_clicks"),
        Input("macd_high", "n_clicks"),
        Input("macd_low", "n_clicks"),
        Input("macd_close", "n_clicks"),
        State("macd_source_column", "label"),
    )
    def update_macd_label(open_clicks, high_clicks, low_clicks, close_clicks, macd_source):
        ctx = dash.callback_context
        if not ctx.triggered:
            return macd_source
        else:
            if ctx.triggered[0]["prop_id"].split(".")[0] == "macd_open":
                return "open"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "macd_high":
                return "high"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "macd_low":
                return "low"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "macd_close":
                return "close"


    @callback(
        Output("cci_source_column", "label"),
        Input("cci_open", "n_clicks"),
        Input("cci_high", "n_clicks"),
        Input("cci_low", "n_clicks"),
        Input("cci_close", "n_clicks"),
        State("cci_source_column", "label"),
    )
    def update_cci_label(open_clicks, high_clicks, low_clicks, close_clicks, cci_source):
        ctx = dash.callback_context
        if not ctx.triggered:
            return cci_source
        else:
            if ctx.triggered[0]["prop_id"].split(".")[0] == "cci_open":
                return "open"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "cci_high":
                return "high"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "cci_low":
                return "low"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "cci_close":
                return "close"


    @callback(
        Output("source_crossover_column", "label"),
        Input("source_open", "n_clicks"),
        Input("source_high", "n_clicks"),
        Input("source_low", "n_clicks"),
        Input("source_close", "n_clicks"),
        State("source_crossover_column", "label"),
    )
    def update_crossover_label(open_clicks, high_clicks, low_clicks, close_clicks, source_crossover):
        ctx = dash.callback_context
        if not ctx.triggered:
            return source_crossover
        else:
            if ctx.triggered[0]["prop_id"].split(".")[0] == "source_open":
                return "open"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "source_high":
                return "high"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "source_low":
                return "low"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "source_close":
                return "close"


    @callback(
        Output("cci_ma_type", "label"),
        Input("sma", "n_clicks"),
        Input("ema", "n_clicks"),
        State("cci_ma_type", "label"),
    )
    def update_cci_ma_type_label(sma_click, ema_click, cci_ma_type):
        ctx = dash.callback_context
        if not ctx.triggered:
            return cci_ma_type
        else:
            if ctx.triggered[0]["prop_id"].split(".")[0] == "sma":
                return "sma"
            if ctx.triggered[0]["prop_id"].split(".")[0] == "ema":
                return "ema"


    @callback(
        Output("interval", "label"),
        Input("1m", "n_clicks"),
        Input("5m", "n_clicks"),
        Input("15m", "n_clicks"),
        Input("30m", "n_clicks"),
        Input("1h", "n_clicks"),
        Input("2h", "n_clicks"),
        Input("4h", "n_clicks"),
        Input("6h", "n_clicks"),
        Input("8h", "n_clicks"),
        Input("12h", "n_clicks"),
        Input("1d", "n_clicks"),
        Input("3d", "n_clicks"),
        Input("1w", "n_clicks"),
        Input("1M", "n_clicks"),
        State("interval", "label"),
    )
    def update_interval_label(m1, m5, m15, m30, h1, h2, h4, h6, h8, h12, d1, d3, w1, M1, interval_label):
        ctx = dash.callback_context
        if not ctx.triggered:
            return interval_label
        else:
            interval = ctx.triggered[0]["prop_id"].split(".")[0]
            return interval