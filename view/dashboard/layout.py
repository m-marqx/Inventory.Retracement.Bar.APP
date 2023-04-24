# %%
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from view.dashboard.components import (
    indicators_parameters_col1,
    indicators_parameters_col2,
    irb_parameters_col1,
    irb_parameters_col2,
    filter_components,
    get_data_components,
    navbar_components,
    header_color,
)

theme = dbc.themes.MORPH
style_sheet = ["assets/style"]

app = dash.Dash(
    "Dashboard",
    external_stylesheets=[theme, style_sheet],
    title="Inventory Retracement Bar",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(
                            """
                            ### Inventory Retracement Bar
                            """
                            ,style={
                                'color': 'rgba(255,255,255,0.85)',
                                "align-items": "center",
                                "justify-content":"center",
                                }
                        ),
                        navbar_components,
                    ],
                    width=True,
                    style={
                        "textAlign": "center",
                        "background-color":header_color,
                        "margin-bottom": "10px"
                    },
                ),
            ],
            align="end",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Get Data",
                            id="data_button",
                            style={"border-radius": "5px"},
                            className="d-grid gap-2 col-6 mx-auto w-100",
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    get_data_components,
                                    style={"display": "flex", "flex-direction": "row"},
                                )
                            ),
                            id="data_collapse",
                            is_open=True,
                            style={"margin-top": "10px"},
                        ),
                        html.Hr(),
                        dbc.Button(
                            "Modify Indicators Parameters",
                            id="operating_button",
                            style={"border-radius": "5px"},
                            className="d-grid gap-2 col-6 mx-auto w-100",
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
                            style={"margin-top": "10px"},
                        ),
                        html.Hr(),
                        dbc.Button(
                            "Modify Strategy Parameters",
                            id="shape_button",
                            style={"border-radius": "5px"},
                            className="d-grid gap-2 col-6 mx-auto w-100",
                        ),
                        dbc.Collapse(
                            style={"margin-top": "10px"},
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
                        html.Hr(),
                        dbc.Button(
                            "Modify Trend Parameters",
                            id="coordinates_button",
                            style={"border-radius": "5px"},
                            className="d-grid gap-2 col-6 mx-auto w-100",
                        ),
                        dbc.Collapse(
                            style={"margin-top": "10px"},
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
                        html.Hr(),
                        dbc.Button(
                            "Run Strategy",
                            id="run_button",
                            style={
                                "margin": "5px",
                                "border-radius": "20px",
                            },
                            color="primary",
                            outline=True,
                            className="d-grid gap-2 col-6 mx-auto w-100",
                        ),
                        html.Hr(),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Div(
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
                                                "title": ""
                                            },
                                            "yaxis": {
                                                "showticklabels": False,
                                                "zeroline": False,
                                                "title": ""
                                            }
                                        }
                                }
                            ),
                            style={
                                "overflow": "hidden",
                                "border-radius": "75px",
                                "background": "linear-gradient(145deg, #d8dde1, #f0f5fa)",
                                "box-shadow": "16px 16px 32px #b4bcc8, -16px -16px 32px #feffff",
                            },
                        ),
                    ],
                    width=9,
                ),
            ]
        ),
        html.Hr(),
    ],
    fluid=True,
    style={"font-family": "Open Sans"},
)
