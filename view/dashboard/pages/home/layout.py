# %%
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from .components import (
    indicators_parameters_col1,
    indicators_parameters_col2,
    irb_parameters_col1,
    irb_parameters_col2,
    filter_components,
)

from view.dashboard.pages.general.components import get_data_components

theme = dbc.themes.MORPH
style_sheet = ["assets/style"]
icons = "https://use.fontawesome.com/releases/v5.15.3/css/all.css"

app = dash.Dash(
    "Dashboard",
    external_stylesheets=[icons, theme, style_sheet],
    title="Inventory Retracement Bar",
)


def layout():
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
                                                "Get Data",
                                                html.I(
                                                    className="fa fa-chevron-up ml-2",
                                                    id="data_icon",
                                                ),
                                            ],
                                            id="data_button",
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
                                            id="data_collapse",
                                            is_open=True,
                                        ),
                                        dbc.Button(
                                            [
                                                "Modify Indicators Parameters",
                                                html.I(
                                                    className="fa fa-chevron-down ml-2",
                                                    id="operating_icon",
                                                ),
                                            ],
                                            id="operating_button",
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
                                                                indicators_parameters_col1,
                                                            ),
                                                            dbc.Col(
                                                                indicators_parameters_col2,
                                                            ),
                                                        ]
                                                    ),
                                                ),
                                            ),
                                            id="operating_collapse",
                                            is_open=False,
                                        ),
                                        dbc.Button(
                                            [
                                                "Modify Strategy Parameters",
                                                html.I(
                                                    className="fa fa-chevron-down ml-2",
                                                    id="shape_icon",
                                                ),
                                            ],
                                            id="shape_button",
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
                                                                irb_parameters_col1,
                                                                style={
                                                                    "display": "flex",
                                                                    "flex-direction": "column",
                                                                },
                                                            ),
                                                            dbc.Col(
                                                                irb_parameters_col2,
                                                                style={
                                                                    "display": "flex",
                                                                    "flex-direction": "column",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                )
                                            ),
                                            id="shape_collapse",
                                            is_open=False,
                                        ),
                                        dbc.Button(
                                            [
                                                "Modify Trend Parameters",
                                                html.I(
                                                    className="fa fa-chevron-down ml-2",
                                                    id="coordinates_icon",
                                                    style={"transformY": "2px"},
                                                ),
                                            ],
                                            id="coordinates_button",
                                            className="d-grid gap-2 col-6 mx-auto w-100",
                                            outline=True,
                                            color="secondary",
                                        ),
                                        dbc.Collapse(
                                            children=dbc.Card(
                                                dbc.CardBody(
                                                    filter_components,
                                                    style={
                                                        "display": "flex",
                                                        "flex-direction": "column",
                                                    },
                                                )
                                            ),
                                            id="coordinates_collapse",
                                            is_open=False,
                                        ),
                                    ],
                                    class_name="d-grid gap-2 col-6 mx-auto w-100 menu-collapse_container",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Run Strategy",
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
                                                "Click Run Button to see results",
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
