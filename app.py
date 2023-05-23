import pathlib
import dash
import pandas as pd
from dash.dependencies import Input, Output, State

from controller.api.klines_api import KlineAPI
from model.utils.utils import SaveDataFrame

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

import numpy as np

from model.backtest import Backtest, BacktestParams

server = app.server

#set language properties
@app.callback(
    Output("lang_selection", "data"),
    Output("home", "href"),
    Output("backtest", "href"),
    Output("en_US_lang", "active"),
    Output("pt_BR_lang", "active"),
    Input("pt_BR_lang", "n_clicks_timestamp"),
    Input("en_US_lang", "n_clicks_timestamp"),
    State("lang_selection", "data")
)
def lang_selection(pt_BR,en_US, lang_selected):

    #This condition will trigger when the user access the app
    if lang_selected == "?lang=pt_BR":
        pt_BR_lang = True
        en_US_lang = False
    else:
        pt_BR_lang = False
        en_US_lang = True

    #This condition will trigger when the user clicks on the language button
    if pt_BR > en_US:
        lang_selection_data = "?lang=pt_BR"
        pt_BR_lang = True
        en_US_lang = False
    else:
        lang_selection_data = "?lang=en_US"
        pt_BR_lang = False
        en_US_lang = True

    home_url = f"/{lang_selection_data}"
    backtest_url = f"/backtest{lang_selection_data}"
    return lang_selection_data, home_url, backtest_url, en_US_lang, pt_BR_lang

#set the backtest params
@app.callback(
    Output("backtest_results", "figure"),
    Output("backtest_text_output", "children"),

    #Get Data
    Input("backtest_run_button", "n_clicks"),
    State("api_types", "value"),
    State("symbol", "value"),
    State("interval", "label"),

    #Indicators Params
    State("min_backtest_ema_length", "value"),
    State("max_backtest_ema_length", "value"),

    #Strategy Params - min values
    State("backtest_min_lowestlow", "value"),
    State("backtest_min_payoff", "value"),
    State("backtest_min_wick_percentage", "value"),

    #Strategy Params - max values
    State("backtest_max_lowestlow", "value"),
    State("backtest_max_payoff", "value"),
    State("backtest_max_wick_percentage", "value"),

    #Trend Params - min values
    State("backtest_min_indicator_macd_histogram_trend_value", "value"),
    State("backtest_min_indicator_cci_trend_value", "value"),

    #Trend Params - max values
    State("backtest_max_indicator_macd_histogram_trend_value", "value"),
    State("backtest_max_indicator_cci_trend_value", "value"),
)
def run_backtest(
    #Get Data
    backtest_run_button,
    api_type,
    symbol,
    interval,

    #Indicators Params
    min_backtest_ema_length,
    max_backtest_ema_length,

    #Strategy Params - min values
    backtest_min_lowestlow,
    backtest_min_payoff,
    backtest_min_wick_percentage,

    #Strategy Params - max values
    backtest_max_lowestlow,
    backtest_max_payoff,
    backtest_max_wick_percentage,

    #Trend Params - min values
    backtest_min_indicator_macd_histogram_trend_value,
    backtest_min_indicator_cci_trend_value,

    #Trend Params - max values
    backtest_max_indicator_macd_histogram_trend_value,
    backtest_max_indicator_cci_trend_value,
):

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    if "backtest_run_button" in ctx.triggered[0]["prop_id"]:
        symbol = symbol.upper()  # Avoid errors when the symbol is in lowercase

        if api_type in ("coin_margined", "mark_price"):
            if symbol.endswith("USD"):
                data_symbol = f"{symbol}_PERP"
            else:
                data_symbol = f"{symbol}"
        else:
            data_symbol = symbol

        data_path = pathlib.Path("model", "data")
        data_name = f"{data_symbol}_{interval}_{api_type}"
        data_file = f"{data_name}.parquet"
        dataframe_path = data_path.joinpath(data_file)

        if dataframe_path.is_file():
            data_frame = pd.read_parquet(dataframe_path)
            kline_api = KlineAPI(data_symbol, interval, api_type)
            data_frame = kline_api.update_data()

        else:
            data_frame = get_data(data_symbol, interval, api_type)

        SaveDataFrame(data_frame).to_parquet(f"{data_name}")

        ema_range = range(min_backtest_ema_length, max_backtest_ema_length + 1)
        lowest_low_range = range(backtest_min_lowestlow, backtest_max_lowestlow + 1)
        payoff_range = range(backtest_min_payoff, backtest_max_payoff + 1)
        wick_percentage_range = np.arange(backtest_min_wick_percentage, backtest_max_wick_percentage + 0.01, 0.01)
        macd_histogram_trend_value_range = range(backtest_min_indicator_macd_histogram_trend_value, backtest_max_indicator_macd_histogram_trend_value + 1)
        cci_trend_value_range = range(backtest_min_indicator_cci_trend_value, backtest_max_indicator_cci_trend_value + 1)

        backtest_params = BacktestParams(
            ema_params={
                "length": list(ema_range),
                "source_column": ["open", "high", "low", "close"]
            },
            irb_params={
                "lowestlow": list(lowest_low_range),
                "payoff": list(payoff_range),
                "ticksize": [0.1],
                "wick_percentage": np.round(wick_percentage_range,2).tolist()
            },
            indicators_params={
                "ema_column": ["open", "high", "low", "close"],
                "macd_histogram_trend_value": list(macd_histogram_trend_value_range),
                "cci_trend_value": list(cci_trend_value_range)
            },
            TrendParamsBacktest={
                "ema" : [True],
                "macd" : [False],
                "cci" : [False],
            },
        )

        backtest = Backtest(data_frame)
        backtest_df = backtest.param_grid_backtest(params=backtest_params)
        backtest_transposed = backtest_df.T

        filtered_df = backtest_transposed.iloc[:,[-1]][backtest_transposed.iloc[:,[-1]] > 0]
        filtered_df.dropna(inplace=True)

        filtered_df_sorted = filtered_df.sort_values(by=filtered_df.columns[-1], ascending=False).index
        final_data_frame = backtest_transposed.loc[filtered_df_sorted].T
        graph_layout = GraphLayout(final_data_frame, data_symbol, interval, api_type)
        fig = graph_layout.grouped_lines()
        text_output = f"Best Result: {final_data_frame.iloc[-1,0]}"

        return fig, text_output

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
def update_label(m1, m5, m15, m30, h1, h2, h4, h6, h8, h12, d1, d3, w1, M1):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Interval"
    else:
        interval = ctx.triggered[0]["prop_id"].split(".")[0]
        return interval


### Callback to make shape parameters menu expand
@app.callback(
    Output("strategy_params_collapse", "is_open"),
    Output("strategy_params_icon", "className"),
    Input("strategy_params_button", "n_clicks"),
    State("strategy_params_collapse", "is_open"),
)
def toggle_strategy_params_collapse(n_clicks, is_open):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if is_open:
        return False, "fa fa-chevron-down ml-2"
    else:
        return True, "fa fa-chevron-up ml-2"


### Callback to make get_data parameters menu expand
@app.callback(
    Output("get_data_collapse", "is_open"),
    Output("get_data_icon", "className"),
    Input("get_data_button", "n_clicks"),
    State("get_data_collapse", "is_open"),
)
def toggle_get_data_collapse(n_clicks, is_open):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if is_open:
        return False, "fa fa-chevron-down"
    else:
        return True, "fa fa-chevron-up"


### Callback to make operating parameters menu expand
@app.callback(
    Output("indicator_params_collapse", "is_open"),
    Output("indicator_params_icon", "className"),
    Input("indicator_params_button", "n_clicks"),
    State("indicator_params_collapse", "is_open"),
)
def toggle_indicator_params_collapse(n_clicks, is_open):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if is_open:
        return False, "fa fa-chevron-down ml-2"
    else:
        return True, "fa fa-chevron-up ml-2"


### Callback to make coordinates menu expand
@app.callback(
    Output("trend_params_collapse", "is_open"),
    Output("trend_params_icon", "className"),
    Input("trend_params_button", "n_clicks"),
    State("trend_params_collapse", "is_open"),
)
def toggle_strategy_params_collapse(n_clicks, is_open):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if is_open:
        return False, "fa fa-chevron-down ml-2"
    else:
        return True, "fa fa-chevron-up ml-2"


@app.callback(
    Output("results", "figure"),
    Output("text_output", "children"),
    Input("run_button", "n_clicks"),
    State("api_types", "value"),
    State("symbol", "value"),
    State("interval", "label"),
    State("ema_source_column", "label"),
    State("ema_length", "value"),
    State("macd_source_column", "label"),
    State("macd_fast_length", "value"),
    State("macd_slow_length", "value"),
    State("macd_signal_length", "value"),
    State("cci_source_column", "label"),
    State("cci_length", "value"),
    State("cci_ma_type", "label"),
    State("cci_constant", "value"),
    State("irb_lowestlow", "value"),
    State("irb_payoff", "value"),
    State("irb_tick_size", "value"),
    State("irb_wick_percentage", "value"),
    State("source_crossover_column", "label"),
    State("indicator_macd_histogram_trend_value", "value"),
    State("indicator_cci_trend_value", "value"),
    State("checklist", "value"),
)
def run_strategy(
    run_button,
    api_type,
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

    if "run_button" in ctx.triggered[0]["prop_id"]:
        symbol = symbol.upper()  # Avoid errors when the symbol is in lowercase

        if api_type in ("coin_margined", "mark_price"):
            if symbol.endswith("USD"):
                data_symbol = f"{symbol}_PERP"
            else:
                data_symbol = f"{symbol}"
        else:
            data_symbol = symbol

        data_path = pathlib.Path("model", "data")
        data_name = f"{data_symbol}_{interval}_{api_type}"
        data_file = f"{data_name}.parquet"
        dataframe_path = data_path.joinpath(data_file)

        if dataframe_path.is_file():
            data_frame = pd.read_parquet(dataframe_path)
            kline_api = KlineAPI(data_symbol, interval, api_type)
            data_frame = kline_api.update_data()

        else:
            data_frame = get_data(data_symbol, interval, api_type)

        SaveDataFrame(data_frame).to_parquet(f"{data_name}")

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
                length=ema_length,
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
                cci=cci_bool,
            ),
        )

        data_frame = builder(data_frame, builder_params)
        graph_layout = GraphLayout(data_frame, data_symbol, interval, api_type)
        fig = graph_layout.plot_cumulative_results()
        text_output = f"Final Result = {data_frame.iloc[-1,-1]:.2f}"
        return fig, text_output


@app.callback(
    Output("home", "active"),
    Output("backtest", "active"),
    Input("home", "n_clicks"),
    Input("backtest", "n_clicks"),
)
def toggle_active_links(home, backtest):
    ctx = dash.callback_context

    pages = {
        "home": True,
        "backtest": False,
    }

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    for i in pages:
        if i == button_id:
            pages[i] = True
        else:
            pages[i] = False

    pages_values = list(pages.values())

    return pages_values[0], pages_values[1]


if __name__ == "__main__":
    app.run(debug=True)
