import dash_bootstrap_components as dbc
from dash import html, dcc
from view.dashboard.pages.lang import en_US, pt_BR

from .utils import (
    intervals,
    api_types,
    result_types,
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
                    dbc.DropdownMenu(
                        children=intervals,
                        label=self.lang["TIMEFRAME"],
                        id="interval",
                    ),
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
    def get_data_components(self):
        return dbc.Row(
            [
                self.api_radio_components,
                dbc.Col(dbc.CardGroup([self.symbol_components])),
                dbc.Col(dbc.CardGroup([self.interval_components])),
            ],
        )


    @property
    def result_type_components(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["RESULT_TYPE"]),
                        width=45,
                        style={"margin": "10px"},
                        class_name="center"
                    ),
                dbc.Col(
                    dbc.RadioItems(
                        result_types(self.lang),
                        id="result_types",
                        input_class_name="btn-check",
                        label_class_name="btn btn-primary",
                        label_checked_class_name="active",
                        inline=True,
                        value="fixed"
                    ),
                    class_name="center"
                )
            ],
            style={"justify-content": "center"},
        )
    @property
    def percentage_component(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Checklist(
                        [{"label": self.lang["USE_PERCENTAGE_RESULTS"], "value": "percentage"}],
                        id="result_percentage",
                        input_class_name="btn-check",
                        label_class_name="btn btn-primary",
                        label_checked_class_name="active",
                        value=[],
                    ),
                    class_name="center"
                )
            ],
            style={"justify-content": "center"},
        )
    @property
    def result_param_first_col(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["INITIAL_CAPITAL"]),
                        width=45,
                        style={"margin-top": "10px"},
                        class_name="center"
                    ),
                dbc.Col(
                    dbc.Input(
                        id="initial_capital_value",
                        value=100_000.0,
                        type="number",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(self.lang["LOSS"]),
                        width=45,
                        style={"margin-top": "10px"},
                        class_name="center"
                    ),
                dbc.Col(
                    dbc.Input(
                        id="loss_result_value",
                        value=-1.0,
                        type="number",
                    ),
                    width=45,
                ),
            ]
        )

    @property
    def result_param_second_col(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["QUANTITY"]),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                dbc.Col(
                    dbc.Input(
                        id="qty_result_value",
                        value=1.0,
                        type="number",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(self.lang["PROFIT"]),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                dbc.Col(
                    dbc.Input(
                        id="gain_result_value",
                        value=2.0,
                        type="number",
                    ),
                    width=45,
                ),
            ]
        )

    @property
    def result_parameters_col1(self):
        return dbc.CardGroup([self.result_param_first_col])

    @property
    def result_parameters_col2(self):
        return dbc.CardGroup([self.result_param_second_col])

    @property
    def result_components(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        self.result_parameters_col1,
                        width=6,
                        style={
                            "display": "flex",
                            "flex-direction": "column",
                        },
                    ),
                    dbc.Col(
                        self.result_parameters_col2,
                        width=6,
                        style={
                            "display": "flex",
                            "flex-direction": "column",
                        },
                    ),
                ]
            ),
            dbc.Row(self.result_type_components),
            dbc.Row(
                self.percentage_component,
                style={"margin-top": "20px"},
            ),
        )
