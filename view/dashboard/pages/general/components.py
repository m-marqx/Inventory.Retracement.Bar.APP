import dash_bootstrap_components as dbc

from .utils import (
    intervals,
    api_types,
)

menu = dbc.Nav(
    [
        dbc.NavItem(
            dbc.NavLink(
                ["DASHBOARD"],
                href="/",
                active=True,
                id="home",
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                ["BACKTEST"],
                href="/backtest",
                active=False,
                id="backtest",
            )
        ),
    ],
    pills=False,
)

navbar_components = dbc.Navbar(
    [
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(menu, id="navbar-collapse", navbar=True),
    ],
    style={"height": "32px"},
)

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
                    value="futures",
                ),
                dbc.Col(id="api_types_output"),
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
