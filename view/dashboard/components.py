import dash_bootstrap_components as dbc
from dash import html, dcc

from .utils import (
    ema_ohlc_items,
    macd_ohlc_items,
    cci_ohlc_items,
    cci_ma_type_items,
    source_ohlc_items,
    indicators_filter,
    intervals,
    api_types,
)


indicators_first_col = dbc.Row(
    [
        dbc.Col(
            dbc.Label("EMA Source Column"),
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
                "EMA Length",
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
                "MACD Fast Length",
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
                "CCI Source Column",
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
                "CCI Length",
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
            dbc.Label("MACD Source Column"),
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
                "MACD Signal Length",
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
                "MACD Slow Length",
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
                "CCI MA Type",
                html_for="cci_ma_type",
                width=45,
            ),
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.DropdownMenu(
                children=cci_ma_type_items,
                label="CCI MA Type",
                id="cci_ma_type",
            ),
            width=45,
        ),
        dbc.Col(
            dbc.Label(
                "CCI Constant",
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
            dbc.Label("IRB Lowest Low"),
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
            dbc.Label("IRB Payoff"),
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
            dbc.Label("IRB Tick Size"),
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
            dbc.Label("IRB Wick Percentage"),
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
            dbc.Label("MACD Bullish Value"),
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
            dbc.Label("CCI Bullish Value"),
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
            dbc.Label("Crossover Price Source"),
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
                "Activate Indicator",
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

symbol_components = dbc.Row(
    [
        # Get Data
        dbc.Col(
            dbc.Label("Symbol"),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.Input(
                id="symbol",
                value="BTCUSD",
                type="text",
            ),
            width=45,
        ),
    ]
)

interval_components = dbc.Row(
    [
        dbc.Col(
            dbc.Label("Interval"),
            width=45,
            style={"margin-top": "10px"},
        ),
        dbc.Col(
            dbc.DropdownMenu(
                children=intervals,
                label="Timeframe",
                id="interval",
            ),
            width=45,
        ),
    ]
)

api_radio_components = dbc.Row(
    [
        dbc.Col(
            [
                dbc.RadioItems(
                    api_types,
                    id="api_types",
                    class_name="btn-group",
                    input_class_name="btn-ghost btn-check",
                    label_class_name="btn-ghost btn btn-primary",
                    label_checked_class_name="active",
                    value='futures',
                ),
                dbc.Col(id='api_types_output'),
            ],
        ),
    ]
)

get_data_components = dbc.Row(
    [
        api_radio_components,
        dbc.Col(dbc.CardGroup([symbol_components])),
        dbc.Col(dbc.CardGroup([interval_components])),
    ],
)

menu = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink(["DASHBOARD"], href="/", active=True)),
    ],
    pills=False,
)

navbar_components = dbc.Navbar(
    [
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(menu, id="navbar-collapse", navbar=True),
    ],
    style={"height":"32px"},
)
