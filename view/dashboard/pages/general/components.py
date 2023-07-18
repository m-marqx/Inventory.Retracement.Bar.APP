import dash_bootstrap_components as dbc
from dash import dcc

from .utils import (
    intervals,
    api_types,
    upload_component,
)

menu = dbc.Nav(
    [
        dbc.NavItem(
            dbc.NavLink(
                "DASHBOARD",
                href="/",
                active=True,
                id="home",
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "BACKTEST",
                href="/backtest",
                active=False,
                id="backtest",
            )
        ),
    ],
)

lang_menu = dbc.Col(
    [
        dbc.NavItem(
            dbc.NavLink(
                "EN",
                href="?lang=en_US",
                id="en_US_lang",
                class_name="nav-link-lang",
                n_clicks_timestamp=1,
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "PT",
                href="?lang=pt_BR",
                id="pt_BR_lang",
                class_name="nav-link-lang",
                n_clicks_timestamp=0,
            )
        ),
        dcc.Store(id="lang_selection", storage_type="local"),
    ],
    id="lang_menu",
    width=5,
)

navbar_components = dbc.Navbar(
    [
        dbc.Collapse(menu, id="navbar-collapse", navbar=True),
        dbc.DropdownMenu(
            lang_menu,
            id="navbar-dropdown",
            nav=True,
            align_end=True,
            label="文/A",
        ),
    ],
)


class GeneralComponents:
    def __init__(self, lang):
        self.lang = lang

    @property
    def symbol_components(self):
        return dbc.Row(
            [
                # Get Data
                dbc.Col(
                    dbc.Label(self.lang["SYMBOL"]),
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

    @property
    def interval_components(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["TIMEFRAME"]),
                    width=45,
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(
                            options=intervals(self.lang),
                            placeholder=self.lang["TIMEFRAME"],
                            id="interval",
                            className="classic_dropdown",
                        ),
                        dbc.Input(
                            class_name="hidden",
                            type="text",
                            id="custom-interval",
                            style={"margin-top": "5px"},
                        )
                    ],
                    id="interval_col",
                    width=45,
                ),
            ]
        )

    @property
    def api_radio_components(self):
        return dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.RadioItems(
                            api_types(self.lang),
                            id="api_types",
                            class_name="btn-group",
                            input_class_name="btn-ghost btn-check",
                            label_class_name="btn-ghost btn btn-primary",
                            label_checked_class_name="active",
                            value="futures",
                        ),
                        dbc.Col(id="api_types_output"),
                    ],
                    class_name="reset-center-row",
                ),
            ]
        )

    @property
    def custom_get_data_components(self):
        return upload_component(
            label=self.lang["UPLOAD_DATA"],
            id_prefix="custom_get_data",
            button_class="w-100"
        )

    @property
    def get_data_components(self):
        return dbc.Row(
            [
                self.api_radio_components,
                dbc.Col(dbc.CardGroup(self.custom_get_data_components), id="custom_get_data"),
                dbc.Col(dbc.CardGroup(self.symbol_components), id="binance_symbol"),
                dbc.Col(dbc.CardGroup(self.interval_components), id="binance_interval"),
            ],
        )
