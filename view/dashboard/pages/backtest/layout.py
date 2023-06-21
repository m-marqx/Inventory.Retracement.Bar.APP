import dash_bootstrap_components as dbc
import psutil

from dash import html, dcc, register_page

from view.dashboard.pages.lang import en_US, pt_BR
from view.dashboard.pages.general.components import GeneralComponents
from view.dashboard.pages.general.utils import MenuCollapse

from .components import BacktestComponents
from .results_components import ResultsComponents

register_page(
    __name__,
    path="/backtest",
    title="Backtest",
    name="backtest",
    description="Rob Hoffman's Inventory Retracement Bar exhaustive search backtest",
)


def layout(lang="en_US"):

    if lang == "pt_BR":
        lang = pt_BR
    else:
        lang = en_US

    backtest_components = BacktestComponents(lang)
    general_components = GeneralComponents(lang)
    results_components = ResultsComponents(lang)

    result_configs_component = MenuCollapse(
        lang=lang,
        label = "MODIFY_RESULT_CONFIGS_BUTTON",
        component=results_components.result_configs,
        id_prefix="result_configs",
    ).components
    result_configs_collapse = result_configs_component[0]
    result_configs_button = result_configs_component[1]

    mem = psutil.virtual_memory()
    available_ram = mem.available

    available_ram_mb = available_ram / 1024 / 1024
    insuficient_available_ram = available_ram_mb < 2048

    if insuficient_available_ram:
        return "Insufficient RAM to run backtest."
    else:
        return [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="backtest_results",
                                            style={
                                                "height": "80vh",
                                            },
                                            figure={
                                                "layout": {
                                                    "paper_bgcolor": "rgba(0,0,0,0)",
                                                    "plot_bgcolor": "rgba(0,0,0,0)",
                                                    "xaxis": {
                                                        "showgrid": False,
                                                        "showticklabels": False,
                                                        "zeroline": False,
                                                        "title": "",
                                                    },
                                                    "yaxis": {
                                                        "showticklabels": False,
                                                        "zeroline": False,
                                                        "gridcolor": "#595959",
                                                        "griddash": "dash",
                                                        "title": "",
                                                        "exponentformat": "none",
                                                    },
                                                }
                                            },
                                            className="graph",
                                        ),
                                    ),
                                ],
                                width=9,
                            ),
                            dbc.Col(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                [
                                                    lang["GET_DATA_BUTTON"],
                                                    html.I(
                                                        className="fa fa-chevron-up ml-2",
                                                        id="get_data_icon",
                                                    ),
                                                ],
                                                id="get_data_button",
                                                className="d-grid gap-2 col-6 mx-auto w-100",
                                                outline=True,
                                                color="secondary",
                                            ),
                                            dbc.Collapse(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        general_components.get_data_components,
                                                        style={
                                                            "display": "flex",
                                                            "flex-direction": "row",
                                                        },
                                                    ),
                                                ),
                                                id="get_data_collapse",
                                                is_open=True,
                                            ),
                                            dbc.Button(
                                                [
                                                    lang[
                                                        "MODIFY_INDICATORS_PARAMETERS_BUTTON"
                                                    ],
                                                    html.I(
                                                        className="fa fa-chevron-down ml-2",
                                                        id="indicator_params_icon",
                                                    ),
                                                ],
                                                id="indicator_params_button",
                                                className="d-grid gap-2 col-6 mx-auto w-100",
                                                outline=True,
                                                color="secondary",
                                            ),
                                            dbc.Collapse(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        backtest_components.indicators_parameters_col1,
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        backtest_components.indicators_parameters_col2,
                                                                        width=6,
                                                                    ),
                                                                ]
                                                            )
                                                        ],
                                                    ),
                                                ),
                                                id="indicator_params_collapse",
                                                is_open=False,
                                            ),
                                            dbc.Button(
                                                [
                                                    lang[
                                                        "MODIFY_STRATEGY_PARAMETERS_BUTTON"
                                                    ],
                                                    html.I(
                                                        className="fa fa-chevron-down ml-2",
                                                        id="strategy_params_icon",
                                                    ),
                                                ],
                                                id="strategy_params_button",
                                                className="d-grid gap-2 col-6 mx-auto w-100",
                                                outline=True,
                                                color="secondary",
                                            ),
                                            dbc.Collapse(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        backtest_components.irb_parameters_col1,
                                                                        style={
                                                                            "display": "flex",
                                                                            "flex-direction": "column",
                                                                        },
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        backtest_components.irb_parameters_col2,
                                                                        style={
                                                                            "display": "flex",
                                                                            "flex-direction": "column",
                                                                        },
                                                                        width=6,
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                    )
                                                ),
                                                id="strategy_params_collapse",
                                                is_open=False,
                                            ),
                                            dbc.Button(
                                                [
                                                    lang[
                                                        "MODIFY_TREND_PARAMETERS_BUTTON"
                                                    ],
                                                    html.I(
                                                        className="fa fa-chevron-down ml-2",
                                                        id="trend_params_icon",
                                                        style={"transformY": "2px"},
                                                    ),
                                                ],
                                                id="trend_params_button",
                                                className="d-grid gap-2 col-6 mx-auto w-100",
                                                outline=True,
                                                color="secondary",
                                            ),
                                            dbc.Collapse(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        backtest_components.filter_components,
                                                    )
                                                ),
                                                id="trend_params_collapse",
                                                is_open=False,
                                            ),
                                            dbc.Button(
                                                [
                                                    lang["MODIFY_RESULT_PARAMETERS_BUTTON"],
                                                    html.I(
                                                        className="fa fa-chevron-down ml-2",
                                                        id="result_params_icon",
                                                        style={"transformY": "2px"},
                                                    ),
                                                ],
                                                id="result_params_button",
                                                className="d-grid gap-2 col-6 mx-auto w-100",
                                                outline=True,
                                                color="secondary",
                                            ),
                                            dbc.Collapse(
                                                dbc.Card(
                                                    [
                                                        dbc.CardBody(
                                                            results_components.result_components,
                                                        ),
                                                        dbc.Card(
                                                            dbc.CardBody([
                                                            result_configs_button,
                                                            ]),
                                                        ),
                                                        result_configs_collapse,
                                                    ]
                                                ),
                                                id="result_params_collapse",
                                                is_open=False,
                                            ),
                                            dbc.Button(
                                                [
                                                    lang[
                                                        "MODIFY_HARDWARE_PARAMETERS_BUTTON"
                                                    ],
                                                    html.I(
                                                        className="fa fa-chevron-down ml-2",
                                                        id="hardware_params_icon",
                                                        style={"transformY": "2px"},
                                                    ),
                                                ],
                                                id="hardware_params_button",
                                                className="d-grid gap-2 col-6 mx-auto w-100",
                                                outline=True,
                                                color="secondary",
                                            ),
                                            dbc.Collapse(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        backtest_components.hardware_components,
                                                    )
                                                ),
                                                id="hardware_params_collapse",
                                                is_open=False,
                                            ),
                                        ],
                                        class_name="d-grid gap-2 col-6 mx-auto w-100 menu-collapse_container",
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                lang["RUN_BACKTEST"],
                                                id="backtest_run_button",
                                                style={
                                                    "margin": "10px",
                                                    "border-radius": "20px",
                                                },
                                                color="primary",
                                                outline=False,
                                                className="d-grid gap-2 col-6 mx-auto w-100",
                                            ),
                                            dbc.Spinner(
                                                html.P(
                                                    lang["EMPTY_RESULT"],
                                                    id="backtest_text_output",
                                                ),
                                                color="primary",
                                                spinner_class_name="spinner-loader",
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        id="backtest_table_component",
                                    )
                                ],
                                width=3,
                            ),
                        ],
                    ),
                ],
                fluid=True,
                style={"font-family": "Open Sans"},
            )
        ]
