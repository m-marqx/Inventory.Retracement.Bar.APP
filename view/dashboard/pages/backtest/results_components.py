import dash_bootstrap_components as dbc

from view.dashboard.pages.general.utils import result_types


class ResultsComponents:
    def __init__(self, lang):
        self.lang = lang

    @property
    def result_type_components(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["RESULT_TYPE"]),
                    width=45,
                    style={"margin": "10px"},
                    class_name="center",
                ),
                dbc.Col(
                    dbc.Checklist(
                        result_types(self.lang),
                        id="backtest_result_types",
                        input_class_name="btn-check",
                        label_class_name="btn btn-primary",
                        label_checked_class_name="active",
                        inline=True,
                        value=["Fixed"],
                    ),
                    class_name="center",
                ),
            ],
            style={"justify-content": "center"},
        )

    @property
    def percentage_component(self):
        return dbc.Col(
            [
                dbc.Col(
                    dbc.Checklist(
                        [
                            {
                                "label": self.lang["USE_PERCENTAGE_RESULTS"],
                                "value": True,
                            }
                        ],
                        id="result_percentage",
                        input_class_name="btn-check",
                        label_class_name="btn btn-primary",
                        label_checked_class_name="active",
                        value=[],
                    ),
                    class_name="center",
                )
            ],
            # style={"justify-content": "center"},
        )

    @property
    def margin_type(self):
        return dbc.Col(
            [
                dbc.Col(
                    dbc.RadioItems(
                        [
                            {"label": "COIN", "value": True},
                            {"label": "USD", "value": False},
                        ],
                        id="result_margin_type",
                        input_class_name="btn-check",
                        label_class_name="btn btn-primary",
                        label_checked_class_name="active",
                        value=False,
                        inline=True,
                    ),
                ),
            ],
        )

    @property
    def plot_type(self):
        return dbc.Col(
            [
                dbc.Col(
                    dbc.Label(self.lang["PLOT_TYPE"]),
                    width=45,
                    style={"margin": "10px"},
                    class_name="center",
                ),
                dbc.Col(
                    dbc.RadioItems(
                        [
                            {"label": self.lang["ALL_LINES"], "value": False},
                            {"label": self.lang["ONLY_POSITIVES"], "value": True},
                        ],
                        id="plot_type",
                        input_class_name="btn-check",
                        label_class_name="btn btn-primary",
                        label_checked_class_name="active",
                        value=True,
                        inline=True,
                    ),
                    class_name="center",
                ),
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
                    class_name="center",
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
                    class_name="center",
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
        )

    @property
    def result_type(self):
        return dbc.Row(
            [
                dbc.Col(self.result_type_components),
                dbc.Row(
                    self.percentage_component,
                    style={"margin-top": "20px"},
                ),
                dbc.Row(
                    self.margin_type,
                    style={"margin-top": "20px"},
                    class_name="hidden",
                    id="backtest_result_margin_type_row",
                ),
                dbc.Col(
                    self.plot_type,
                    style={"margin-top": "20px"},
                ),
            ]
        )
