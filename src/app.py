import dash
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State

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

from view.dashboard.layout import app

from view.dashboard.utils import (
    BuilderParams,
    get_data,
    builder,
)

from view.dashboard.graph import GraphLayout

server = app.server

data_frame = pd.DataFrame()
fig = px.line(data_frame).update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        title="",
    ),
    yaxis=dict(
        showgrid=True,
        showticklabels=False,
        zeroline=False,
        title="",
    )
)

# update the DropdownMenu items
@app.callback(
    Output("ema_source_column", "label"),
    Input("ema_open", "n_clicks"),
    Input("ema_high", "n_clicks"),
    Input("ema_low", "n_clicks"),
    Input("ema_close", "n_clicks"),
)
def update_label(open_clicks, high_clicks, low_clicks, close_clicks):
    global ema_button_id
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Column"
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


@app.callback(
    Output("macd_source_column", "label"),
    Input("macd_open", "n_clicks"),
    Input("macd_high", "n_clicks"),
    Input("macd_low", "n_clicks"),
    Input("macd_close", "n_clicks"),
)
def update_label(open_clicks, high_clicks, low_clicks, close_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Column"
    else:
        if ctx.triggered[0]["prop_id"].split(".")[0] == "macd_open":
            return "open"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "macd_high":
            return "high"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "macd_low":
            return "low"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "macd_close":
            return "close"


@app.callback(
    Output("cci_source_column", "label"),
    Input("cci_open", "n_clicks"),
    Input("cci_high", "n_clicks"),
    Input("cci_low", "n_clicks"),
    Input("cci_close", "n_clicks"),
)
def update_label(open_clicks, high_clicks, low_clicks, close_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Column"
    else:
        if ctx.triggered[0]["prop_id"].split(".")[0] == "cci_open":
            return "open"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "cci_high":
            return "high"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "cci_low":
            return "low"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "cci_close":
            return "close"


@app.callback(
    Output("source_crossover_column", "label"),
    Input("source_open", "n_clicks"),
    Input("source_high", "n_clicks"),
    Input("source_low", "n_clicks"),
    Input("source_close", "n_clicks"),
)
def update_label(open_clicks, high_clicks, low_clicks, close_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Column"
    else:
        if ctx.triggered[0]["prop_id"].split(".")[0] == "source_open":
            return "open"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "source_high":
            return "high"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "source_low":
            return "low"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "source_close":
            return "close"


@app.callback(
    Output("cci_ma_type", "label"),
    Input("sma", "n_clicks"),
    Input("ema", "n_clicks"),
)
def update_label(sma_click, ema_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Column"
    else:
        if ctx.triggered[0]["prop_id"].split(".")[0] == "sma":
            return "sma"
        if ctx.triggered[0]["prop_id"].split(".")[0] == "ema":
            return "ema"


@app.callback(
    Output("interval", "label"),
    [
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
    ],
)
def update_label(m1,m5,m15,m30,h1,h2,h4,h6,h8,h12,d1,d3,w1,M1):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Interval"
    else:
        interval = ctx.triggered[0]["prop_id"].split(".")[0]
        return interval


### Callback to make shape parameters menu expand
@app.callback(
    Output("shape_collapse", "is_open"),
    [Input("shape_button", "n_clicks")],
    [State("shape_collapse", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make get_data parameters menu expand
@app.callback(
    Output("data_collapse", "is_open"),
    [Input("data_button", "n_clicks")],
    [State("data_collapse", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make operating parameters menu expand
@app.callback(
    Output("operating_collapse", "is_open"),
    [Input("operating_button", "n_clicks")],
    [State("operating_collapse", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make coordinates menu expand
@app.callback(
    Output("coordinates_collapse", "is_open"),
    [Input("coordinates_button", "n_clicks")],
    [State("coordinates_collapse", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


data_params = []


@app.callback(
    Output("results", "figure"),
    [
        Input("run_button", "n_clicks"),
        Input("symbol", "value"),
        Input("interval", "label"),
        Input("ema_source_column", "label"),
        Input("ema_length", "value"),
        Input("macd_source_column", "label"),
        Input("macd_fast_length", "value"),
        Input("macd_slow_length", "value"),
        Input("macd_signal_length", "value"),
        Input("cci_source_column", "label"),
        Input("cci_length", "value"),
        Input("cci_ma_type", "label"),
        Input("cci_constant", "value"),
        Input("irb_lowestlow", "value"),
        Input("irb_payoff", "value"),
        Input("irb_tick_size", "value"),
        Input("irb_wick_percentage", "value"),
        Input("source_crossover_column", "label"),
        Input("indicator_macd_histogram_trend_value", "value"),
        Input("indicator_cci_trend_value", "value"),
        Input("checklist", "value"),
    ],
)
def analyze(
    run_button,
    symbol,
    interval,
    ema_source_column,
    ema_length,
    macd_source_column,
    macd_fast_length,
    macd_slow_length,
    macd_signal_length,
    cci_source_column,
    cci_length,
    cci_ma_type,
    cci_constant,
    irb_lowestlow,
    irb_payoff,
    irb_tick_size,
    irb_wick_percentage,
    source_crossover_column,
    indicator_macd_histogram_trend_value,
    indicator_cci_trend_value,
    checklist,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    global data_params
    global fig
    global data_frame

    if "run_button" in ctx.triggered[0]["prop_id"]:

        data_params.append(symbol + interval)
        empty_data_frame = data_frame.shape[0] == 0
        is_same_data_params = (
            len(data_params) > 1 and data_params[-1] == data_params[-2]
        )
        is_different_data_params = (
            len(data_params) > 1 and data_params[-1] != data_params[-2]
        )
        if empty_data_frame:
            data_frame = get_data(symbol, interval)
        if is_same_data_params:
            data_params = [data_params[-1]]
            print(f"{data_params} is same params")
        if is_different_data_params:
            data_frame = get_data(symbol, interval)
            data_params = [data_params[-1]]
            print(f"{data_params} will be reset")

        ema_bool = False
        macd_bool = False
        cci_bool = False

        if checklist is not None:
            for bool_param in checklist:
                if bool_param == "ema":
                    ema_bool = True
                if bool_param == "macd":
                    macd_bool = True
                if bool_param == "cci":
                    cci_bool = True

        builder_params = BuilderParams(
            ema_params=EmaParams(
                source_column=ema_source_column,
                length=ema_length
            ),
            macd_params=MACDParams(
                source_column=macd_source_column,
                fast_length=macd_fast_length,
                slow_length=macd_slow_length,
                signal_length=macd_signal_length,
            ),
            cci_params=CCIParams(
                source_column=cci_source_column,
                length=cci_length,
                ma_type=cci_ma_type,
                constant=cci_constant,
            ),
            irb_params=IrbParams(
                lowestlow=irb_lowestlow,
                payoff=irb_payoff,
                tick_size=irb_tick_size,
                wick_percentage=irb_wick_percentage,
            ),
            indicator_params=IndicatorsParams(
                ema_column=source_crossover_column,
                macd_histogram_trend_value=indicator_macd_histogram_trend_value,
                cci_trend_value=indicator_cci_trend_value,
            ),
            trend_params=TrendParams(
                ema=ema_bool,
                macd=macd_bool,
                cci=cci_bool
            ),
        )

        data_frame = builder(data_frame, builder_params)
        fig = GraphLayout(data_frame).plot_cumulative_results(symbol, interval)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
