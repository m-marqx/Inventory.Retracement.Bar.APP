import dash_bootstrap_components as dbc
import torch
import psutil


class BacktestComponents:
    def __init__(self, lang):

        self.lang = lang
        self.gpu_count = torch.cuda.device_count()
        self.disable_gpu_input = self.gpu_count == 0

    @property
    def indicators_first_col(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Label(
                            self.lang["MIN_EMA_LENGTH"],
                        ),
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="min_backtest_ema_length",
                            value=20,
                            type="number",
                        ),
                        width=45,
                    ),
                ],
            )
        )

    @property
    def indicators_second_col(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Label(
                            self.lang["MAX_EMA_LENGTH"],
                        ),
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="max_backtest_ema_length",
                            value=21,
                            type="number",
                        ),
                        width=45,
                    ),
                ]
            )
        )

    @property
    def indicators_parameters_col1(self):
        return dbc.CardGroup([self.indicators_first_col])

    @property
    def indicators_parameters_col2(self):
        return dbc.CardGroup([self.indicators_second_col])


    @property
    def irb_components_first_col(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Label(
                            self.lang["MIN_LOWEST_LOW_LENGTH"]
                            ),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_min_lowestlow",
                            value=1,
                            type="number",
                        ),
                        width=45,
                    ),
                    dbc.Col(
                        dbc.Label(
                            self.lang["MIN_PAYOFF"]
                            ),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_min_payoff",
                            value=2,
                            type="number",
                        ),
                        width=45,
                    ),
                    dbc.Col(
                        dbc.Label(
                            self.lang["MIN_WICK_PERCENTAGE"],
                            ),
                        width=45,
                        style={
                            "margin-top": "10px",
                        },
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_min_wick_percentage",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.01,
                            type="number",
                        ),
                        width=45,
                    ),
                ]
            )
        )

    @property
    def irb_components_second_col(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Label(
                            self.lang["MAX_LOWEST_LOW_LENGTH"]
                            ),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_max_lowestlow",
                            value=1,
                            type="number",
                        ),
                        width=45,
                    ),
                    dbc.Col(
                        dbc.Label(
                            self.lang["MAX_PAYOFF"]
                            ),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_max_payoff",
                            value=2,
                            type="number",
                        ),
                        width=45,
                    ),
                    dbc.Col(
                        dbc.Label(
                            self.lang["MAX_WICK_PERCENTAGE"],
                            ),
                        width=45,
                        style={
                            "margin-top": "10px",
                        },
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_max_wick_percentage",
                            min=0,
                            max=1,
                            step=0.01,
                            value=1,
                            type="number",
                        ),
                        width=45,
                    ),
                ]
            )
        )

    @property
    def irb_parameters_col1(self):
        return dbc.CardGroup([self.irb_components_first_col])

    @property
    def irb_parameters_col2(self):
        return dbc.CardGroup([self.irb_components_second_col])

    @property
    def macd_cci_min_values_components(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Label(
                            self.lang["MIN_MACD_BULLISH_VALUE"]
                            ),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_min_indicator_macd_histogram_trend_value",
                            value=0,
                            type="number",
                        ),
                        width=45,
                    ),
                    dbc.Col(
                        dbc.Label(
                            self.lang["MIN_CCI_BULLISH_VALUE"]
                            ),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_min_indicator_cci_trend_value",
                            value=0,
                            type="number",
                        ),
                        width=45,
                    ),
                ]
            )
        )

    @property
    def cci_bullish_value_components(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Label(
                            self.lang["MAX_MACD_BULLISH_VALUE"]
                            ),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_max_indicator_macd_histogram_trend_value",
                            value=0,
                            type="number",
                        ),
                        width=45,
                    ),
                    dbc.Col(
                        dbc.Label(
                            self.lang["MAX_CCI_BULLISH_VALUE"]
                            ),
                        width=45,
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="backtest_max_indicator_cci_trend_value",
                            value=0,
                            type="number",
                        ),
                        width=45,
                    ),
                ]
            )
        )

    @property
    def filter_components(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.CardGroup([self.macd_cci_min_values_components]),
                        width=6,
                        ),
                    dbc.Col(
                        dbc.CardGroup([self.cci_bullish_value_components]),
                        width=6,
                        ),
                ]
            )
        )

    @property
    def hardware_radio_components_label(self):
        return (
            dbc.Row(
                [
                    dbc.Label(
                        self.lang["HARDWARE_TYPE"],
                        width=45,
                        style={"margin-top": "10px"},
                        className="center",
                    ),
                ],
                className="center",
            )
        )

    @property
    def hardware_radio_components_input(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.RadioItems(
                                [
                                    {"label": "CPU", "value": "CPU"},
                                    {
                                        "label": "GPU",
                                        "value": "GPU",
                                        "disabled": self.disable_gpu_input
                                    },
                                ],
                                id="hardware_types",
                                class_name="btn-group",
                                input_class_name="btn-ghost btn-check",
                                label_class_name="btn-ghost btn btn-primary",
                                label_checked_class_name="active",
                                value="CPU",
                            ),
                            dbc.Col(id="hardware_types_output"),
                        ],
                        class_name="center",
                    ),
                ]
            )
        )

    @property
    def hardware_gpu_label(self):
        return (
            dbc.Row(
                dbc.Col(
                    dbc.Label(
                        self.lang["GPU_NUMBER"]
                    ),
                    width=45,
                    style={"margin-top": "10px"},
                )
            )
        )

    @property
    def hardware_gpu_input(self):
        return (
            dbc.Row(
                dbc.Col(
                    dbc.Input(
                        id="backtest_gpu_number",
                        value=self.gpu_count,
                        type="number",
                        disabled=self.disable_gpu_input,
                    ),
                    width=45,
                ),
            )
        )

    @property
    def hardware_cpu_cores_label(self):
        return (
            dbc.Row(
                dbc.Col(
                    dbc.Label(
                        self.lang["CPU_CORES_NUMBER"]
                    ),
                    width=45,
                    style={"margin-top": "10px"},
                )
            )
        )

    @property
    def hardware_cpu_cores_input(self):
        return (
            dbc.Row(
                dbc.Col(
                    dbc.Input(
                        id="backtest_cpu_cores_number",
                        value=psutil.cpu_count(),
                        type="number",
                    ),
                    width=45,
                ),
            )
        )

    @property
    def hardware_worker_label(self):
        return (
            dbc.Row(
                dbc.Col(
                    dbc.Label(
                        self.lang["GPU_WORKERS_NUMBERS"]
                    ),
                    width=45,
                    style={"margin-top": "10px"},
                )
            )
        )

    @property
    def hardware_worker_input(self):
        return (
            dbc.Row(
                dbc.Col(
                    dbc.Input(
                        id="backtest_workers_number",
                        value=4 if self.gpu_count > 0 else 0,
                        type="number",
                        disabled=self.disable_gpu_input,
                    ),
                    width=45,
                ),
            )
        )

    @property
    def hardware_components(self):
        return (
            dbc.Row(
                [
                    dbc.Col(dbc.CardGroup(self.hardware_radio_components_label)),
                    dbc.Col(dbc.CardGroup(self.hardware_radio_components_input)),
                ],
                class_name="center-row",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.CardGroup(self.hardware_cpu_cores_label)),
                    dbc.Col(dbc.CardGroup(self.hardware_cpu_cores_input)),
                ],
                class_name="center-row",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.CardGroup(self.hardware_gpu_label)),
                    dbc.Col(dbc.CardGroup(self.hardware_gpu_input)),
                ],
                class_name="center-row",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.CardGroup(self.hardware_worker_label)),
                    dbc.Col(dbc.CardGroup(self.hardware_worker_input)),
                ],
                class_name="center-row",
            ),
        )
