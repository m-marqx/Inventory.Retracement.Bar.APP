from dash import dcc, html, register_page
import dash_bootstrap_components as dbc
from view.dashboard.pages.lang import en_US, pt_BR

from view.dashboard.pages import LayoutMenuCollapse
from view.dashboard.pages.ml.components import MLComponents


register_page(
    __name__,
    path="/ml",
    title="Machine Learning",
    name="Machine Learning",
    description="Machine Learning simple backtest.",
)


def layout(lang="en_US"):
    if lang == "en_US":
        lang = en_US
    elif lang == "pt_BR":
        lang = pt_BR

    ml_components = MLComponents(lang)

    return [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="ml_results",
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
                            width=8,
                        ),
                        dbc.Col(
                            [
                                ml_components.collapse_preset_settings,
                                ml_components.indicators_settings,
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Button(
                                                    lang["RUN_MODEL"],
                                                    id="run_model",
                                                    style={
                                                        "margin-top": "20px",
                                                        "border-top-left-radius": "20px",
                                                        "border-bottom-left-radius": "20px",
                                                        "border-top-right-radius": "0px",
                                                        "border-bottom-right-radius": "0px",
                                                    },
                                                    color="primary",
                                                    outline=False,
                                                    className="w-50",
                                                ),
                                                dbc.Button(
                                                    lang["CANCEL_MODEL"],
                                                    id="cancel_model",
                                                    style={
                                                        "margin-top": "20px",
                                                        "border-top-left-radius": "0px",
                                                        "border-bottom-left-radius": "0px",
                                                        "border-top-right-radius": "20px",
                                                        "border-bottom-right-radius": "20px",
                                                    },
                                                    color="primary",
                                                    disabled=True,
                                                    outline=False,
                                                    className="w-50",
                                                ),
                                                html.Progress(
                                                    id="progress_bar",
                                                    value="0",
                                                    max="100",
                                                    className="progress-bar-model",
                                                    style={
                                                        "margin-top": "10px"
                                                    }
                                                ),
                                            ],
                                            style={
                                                "margin-left": "auto",
                                                "margin-right": "auto",
                                            },
                                        ),
                                        dbc.Spinner(
                                            [
                                                html.P(
                                                    lang["EMPTY_RESULT"],
                                                    id="model_text_output",
                                                    style={
                                                        "margin-top": "10px",
                                                        "border-radius": "20px",
                                                        "display": "flex",
                                                        "flex-wrap": "wrap",
                                                        "align-content": "flex-start",
                                                        "justify-content": "center",
                                                    }),
                                                html.P(
                                                    id="new_signal_output",
                                                    style={
                                                        "margin-top": "10px",
                                                        "border-radius": "20px",
                                                        "display": "flex",
                                                        "flex-wrap": "wrap",
                                                        "align-content": "flex-start",
                                                        "justify-content": "center",
                                                    }),
                                            ],
                                            id="text_model_spinner",
                                            color="primary",
                                            spinner_class_name="spinner-loader",
                                            spinner_style={
                                                    "margin-top": "10px",
                                            },
                                        ),
                                    ]
                                ),
                                dbc.Col(id="table_container"),
                            ],
                            width=4,
                        ),
                    ],
                ),
            ],
            fluid=True,
            style={"font-family": "Open Sans"},
        )
    ]
