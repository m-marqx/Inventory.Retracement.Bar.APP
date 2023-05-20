import dash_bootstrap_components as dbc
from dash import html, dcc
from view.dashboard.pages.lang.en_US import lang

from .utils import (
    ema_ohlc_items,
    macd_ohlc_items,
    cci_ohlc_items,
    cci_ma_type_items,
    source_ohlc_items,
    indicators_filter,
)


indicators_first_col = dbc.Row(
    [
        dbc.Col(
            dbc.Label(lang["EMA_SOURCE_COLUMN"]),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.DropdownMenu(
                children=ema_ohlc_items,
                label="Column",
                id="ema_source_column",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(
                lang["EMA_LENGTH"],
                html_for="ema_length",
                width=45,
            ),
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="ema_length",
                value=20,
                type="number",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(
                lang["MACD_FAST_LENGTH"],
                html_for="macd_fast_length",
                width=45,
            ),
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="macd_fast_length",
                value=45,
                type="number",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(
                lang["CCI_SOURCE_COLUMN"],
                html_for="cci_source_column",
                width=45,
            ),
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.DropdownMenu(
                children=cci_ohlc_items,
                label="Column",
                id="cci_source_column",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(
                lang["CCI_LENGTH"],
                html_for="cci_length",
                width=45,
            ),
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="cci_length",
                value=20,
                type="number",
            ),
            width=45,
        ),
    ],
)
indicators_second_col = dbc.Row(
    [
        dbc.Col(
            dbc.Label(lang["MACD_SOURCE_COLUMN"]),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.DropdownMenu(
                children=macd_ohlc_items,
                label="Column",
                id="macd_source_column",
            ),
            width=45,
        ),
        dcc.Store(id="macd_source_column_value"),
        dbc.Col(
            dbc.Label(
                lang["MACD_SIGNAL_LENGTH"],
                html_for="macd_signal_length",
                width=45,
            ),
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="macd_signal_length",
                value=8,
                type="number",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(
                lang["MACD_SLOW_LENGTH"],
                html_for="macd_slow_length",
                width=45,
            ),
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="macd_slow_length",
                value=100,
                type="number",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(
                lang["CCI_MA_TYPE"],
                html_for="cci_ma_type",
                width=45,
            ),
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.DropdownMenu(
                children=cci_ma_type_items,
                label=lang["CCI_MA_TYPE"],
                id="cci_ma_type",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(
                lang["CCI_CONSTANT"],
                html_for="cci_constant",
                width=45,
            ),
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="cci_constant",
                value=0.015,
                type="number",
            ),
            width=45,
        ),
    ]
)

indicators_parameters_col1 = dbc.CardGroup([indicators_first_col])
indicators_parameters_col2 = dbc.CardGroup([indicators_second_col])

irb_components_first_col = dbc.Row(
    [
        dbc.Col(
            dbc.Label(lang["LOWEST_LOW"]),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="irb_lowestlow",
                value=1,
                type="number",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(lang["PAYOFF"]),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="irb_payoff",
                value=2,
                type="number",
            ),
            width=45,
        ),
    ]
)
irb_components_second_col = dbc.Row(
    [
        dbc.Col(
            dbc.Label(lang["TICK_SIZE"]),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="irb_tick_size",
                value=0.1,
                type="number",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(lang["WICK_PERCENTAGE"]),
            width=45,
            style={
                "margin-top": "10px",
            },
        ),
        dbc.Col(
            dbc.Input(
                id="irb_wick_percentage",
                value=0.45,
                type="number",
            ),
            width=45,
        ),
    ]
)


irb_parameters_col1 = dbc.CardGroup([irb_components_first_col])
irb_parameters_col2 = dbc.CardGroup([irb_components_second_col])

macd_bullish_value_components = dbc.Row(
    [
        dbc.Col(
            dbc.Label(lang["MACD_BULLISH_VALUE"]),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="indicator_macd_histogram_trend_value",
                value=0,
                type="number",
            ),
            width=45,
        ),
    ]
)

cci_bullish_value_components = dbc.Row(
    [
        dbc.Col(
            dbc.Label(lang["CCI_BULLISH_VALUE"]),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="indicator_cci_trend_value",
                value=0,
                type="number",
            ),
            width=45,
        ),
    ]
)

macd_cci_components = dbc.Row(
    [
        dbc.Col(dbc.CardGroup([macd_bullish_value_components])),
        dbc.Col(dbc.CardGroup([cci_bullish_value_components])),
    ]
)

filter_components = dbc.Row(
    [
        dbc.Col(
            dbc.Label(lang["CROSSOVER_PRICE_SOURCE"]),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.DropdownMenu(
                children=source_ohlc_items,
                label="Column",
                id="source_crossover_column",
            ),
            width=45,
        ),
        macd_cci_components,
        dbc.Col(
            html.Label(
                lang["ACTIVATE_INDICATOR"],
                style={
                    "font-weight": "bold",
                    "font-size": "20px",
                    "margin-top": "10px",
                },
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Checklist(
                indicators_filter,
                id="checklist",
                input_class_name="btn-check",
                label_class_name="btn btn-primary",
                label_checked_class_name="active",
                inline=True,
            )
        ),
    ]
)

filter_components_col1 = dbc.CardGroup([filter_components])