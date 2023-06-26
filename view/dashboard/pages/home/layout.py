from dash import dcc, html, register_page
import dash_bootstrap_components as dbc
from view.dashboard.pages.lang import en_US, pt_BR

from view.dashboard.pages.general.collapse_menus import LayoutMenuCollapse


theme = dbc.themes.MORPH
style_sheet = ["assets/style"]
icons = "https://use.fontawesome.com/releases/v5.15.3/css/all.css"

register_page(
    __name__,
    path="/",
    title="Main Page",
    name="Main Page",
    description="Rob Hoffman's Inventory Retracement Bar simple backtest.",
)


def layout(lang="en_US"):
    if lang == "en_US":
        lang = en_US
    elif lang == "pt_BR":
        lang = pt_BR

    main_page_collapse = LayoutMenuCollapse(lang, "home")
    get_data_component = main_page_collapse.get_data_component
    parameters_component = main_page_collapse.parameters_component
    strategy_component = main_page_collapse.strategy_component
    trend_component = main_page_collapse.trend_component
    result_parameters_component = main_page_collapse.result_parameters_component

    return [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="results",
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
                                    ],
                                    class_name="d-grid gap-2 col-6 mx-auto w-100 menu-collapse_container",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            lang["RUN_STRATEGY"],
                                            id="run_button",
                                            style={
                                                "margin-top": "10px",
                                                "border-radius": "20px",
                                            },
                                            color="primary",
                                            outline=False,
                                            className="d-grid gap-2 col-6 mx-auto w-100",
                                        ),
                                        dbc.Spinner(
                                            html.P(
                                                lang["EMPTY_RESULT"],
                                                id="text_output",
                                            ),
                                            color="primary",
                                            spinner_class_name="spinner-loader",
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    id="table_container"
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
