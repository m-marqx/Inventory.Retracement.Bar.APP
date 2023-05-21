# %%
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from view.dashboard.pages.general.components import get_data_components
from view.dashboard.pages.lang import en_US, pt_BR

from .components import MainPageComponents


theme = dbc.themes.MORPH
style_sheet = ["assets/style"]
icons = "https://use.fontawesome.com/releases/v5.15.3/css/all.css"

app = dash.Dash(
    "Dashboard",
    external_stylesheets=[icons, theme, style_sheet],
    title="Inventory Retracement Bar",
)


def main_page(lang="en_US"):
    if lang == "en_US":
        lang = en_US
    elif lang == "pt_BR":
        lang = pt_BR

    main_page_components = MainPageComponents(lang)

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
                                                    get_data_components,
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
                                                lang["MODIFY_INDICATORS_PARAMETERS_BUTTON"],
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
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                main_page_components.indicators_parameters_col1,
                                                            ),
                                                            dbc.Col(
                                                                main_page_components.indicators_parameters_col2,
                                                            ),
                                                        ]
                                                    ),
                                                ),
                                            ),
                                            id="indicator_params_collapse",
                                            is_open=False,
                                        ),
                                        dbc.Button(
                                            [
                                                lang["MODIFY_STRATEGY_PARAMETERS_BUTTON"],
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
                                            children=dbc.Card(
                                                dbc.CardBody(
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                main_page_components.irb_parameters_col1,
                                                                style={
                                                                    "display": "flex",
                                                                    "flex-direction": "column",
                                                                },
                                                            ),
                                                            dbc.Col(
                                                                main_page_components.irb_parameters_col2,
                                                                style={
                                                                    "display": "flex",
                                                                    "flex-direction": "column",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                )
                                            ),
                                            id="strategy_params_collapse",
                                            is_open=False,
                                        ),
                                        dbc.Button(
                                            [
                                                lang["MODIFY_TREND_PARAMETERS_BUTTON"],
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
                                            children=dbc.Card(
                                                dbc.CardBody(
                                                    main_page_components.filter_components,
                                                    style={
                                                        "display": "flex",
                                                        "flex-direction": "column",
                                                    },
                                                )
                                            ),
                                            id="trend_params_collapse",
                                            is_open=False,
                                        ),
                                    ],
                                    class_name="d-grid gap-2 col-6 mx-auto w-100 menu-collapse_container",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            lang["RUN_STRATEGY"],
                                            id="run_button",
                                            style={
                                                "margin": "5px",
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
                            ],
                        ),
                    ],
                ),
            ],
            fluid=True,
            style={"font-family": "Open Sans"},
        )
    ]
