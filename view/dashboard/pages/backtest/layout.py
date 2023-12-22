import psutil
from dash import html, dcc, register_page
import dash_bootstrap_components as dbc

from view.dashboard.pages.lang import en_US, pt_BR
from view.dashboard.pages import LayoutMenuCollapse

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

    backtest_menu_collapse = LayoutMenuCollapse(lang, "Backtest")

    get_data_component = backtest_menu_collapse.get_data_component
    parameters_component = backtest_menu_collapse.parameters_component
    strategy_component = backtest_menu_collapse.strategy_component
    trend_component = backtest_menu_collapse.trend_component
    result_parameters_component = backtest_menu_collapse.result_parameters_component
    hardware_component = backtest_menu_collapse.hardware_component

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
                                            get_data_component,
                                            parameters_component,
                                            strategy_component,
                                            trend_component,
                                            result_parameters_component,
                                            hardware_component,
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
                                    ),
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
