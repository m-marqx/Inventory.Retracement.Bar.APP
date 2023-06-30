import dash_bootstrap_components as dbc

from view.dashboard.pages.general.utils import result_types
from .utils import DropdownMenuItems


class MainPageComponents:
    def __init__(self, lang):
        self.lang = lang
        self.dropdown_menu_item = DropdownMenuItems(lang)

    @property
    def indicators_first_col(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["EMA_SOURCE_PRICE"]),
                    width=45,
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.DropdownMenu(
                        children=self.dropdown_menu_item.ema_ohlc_items,
                        label=self.lang["SOURCE"],
                        id="ema_source_column",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(
                        self.lang["EMA_LENGTH"],
                        html_for="ema_length",
                        width=45,
                    ),
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="ema_length",
                        value=20,
                        type="number",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(
                        self.lang["MACD_FAST_LENGTH"],
                        html_for="macd_fast_length",
                        width=45,
                    ),
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="macd_fast_length",
                        value=45,
                        type="number",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(
                        self.lang["CCI_SOURCE_PRICE"],
                        html_for="cci_source_column",
                        width=45,
                    ),
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.DropdownMenu(
                        children=self.dropdown_menu_item.cci_ohlc_items,
                        label=self.lang["SOURCE"],
                        id="cci_source_column",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(
                        self.lang["CCI_LENGTH"],
                        html_for="cci_length",
                        width=45,
                    ),
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="cci_length",
                        value=20,
                        type="number",
                    ),
                    width=45,
                ),
            ],
        )

    @property
    def indicators_second_col(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["MACD_SOURCE_PRICE"]),
                    width=45,
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.DropdownMenu(
                        children=self.dropdown_menu_item.macd_ohlc_items,
                        label=self.lang["SOURCE"],
                        id="macd_source_column",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(
                        self.lang["MACD_SIGNAL_LENGTH"],
                        html_for="macd_signal_length",
                        width=45,
                    ),
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="macd_signal_length",
                        value=8,
                        type="number",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(
                        self.lang["MACD_SLOW_LENGTH"],
                        html_for="macd_slow_length",
                        width=45,
                    ),
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="macd_slow_length",
                        value=100,
                        type="number",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(
                        self.lang["CCI_MA_TYPE"],
                        html_for="cci_ma_type",
                        width=45,
                    ),
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.DropdownMenu(
                        children=self.dropdown_menu_item.cci_ma_type_items,
                        label=self.lang["CCI_MA_TYPE"],
                        id="cci_ma_type",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(
                        self.lang["CCI_CONSTANT"],
                        html_for="cci_constant",
                        width=45,
                    ),
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="cci_constant",
                        value=0.015,
                        type="number",
                    ),
                    width=45,
                ),
            ]
        )

    @property
    def indicators_parameters_col1(self):
        return dbc.CardGroup([self.indicators_first_col])

    @property
    def indicators_parameters_col2(self):
        return dbc.CardGroup([self.indicators_second_col])

    @property
    def irb_components_first_col(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["LOWEST_LOW"]),
                    width=45,
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="irb_lowestlow",
                        value=1,
                        type="number",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(self.lang["PAYOFF"]),
                    width=45,
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="irb_payoff",
                        value=2,
                        type="number",
                    ),
                    width=45,
                ),
            ]
        )

    @property
    def irb_components_second_col(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["TICK_SIZE"]),
                    width=45,
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="irb_tick_size",
                        value=0.1,
                        type="number",
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Label(self.lang["WICK_PERCENTAGE"]),
                    width=45,
                    style={
                        "margin-top": "10px",
                    },
                ),
                dbc.Col(
                    dbc.Input(
                        id="irb_wick_percentage",
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.45,
                        type="number",
                    ),
                    width=45,
                ),
            ]
        )

    @property
    def irb_parameters_col1(self):
        return dbc.CardGroup(self.irb_components_first_col)

    @property
    def irb_parameters_col2(self):
        return dbc.CardGroup(self.irb_components_second_col)

    @property
    def trend_indicators_label1(self):
        return (
            dbc.Row(
                dbc.Label(
                    self.lang["ACTIVATE_INDICATOR"],
                    width=45,
                    style={"margin-top": "10px"},
                    class_name="center",
                ),
                class_name="center",
            ),
        )

    @property
    def trend_indicators_label2(self):
        return (
            dbc.Row(
                dbc.Label(
                    self.lang["CROSSOVER_PRICE_SOURCE"],
                    width=45,
                    style={"margin-top": "10px"},
                    class_name="center",
                ),
                class_name="center",
            ),
        )

    @property
    def trend_indicators_label3(self):
        return (
            dbc.Row(
                dbc.Label(
                    self.lang["MACD_BULLISH_VALUE"],
                    width=45,
                    style={"margin-top": "10px"},
                    class_name="center",
                ),
                class_name="center",
            ),
        )

    @property
    def trend_indicators_label4(self):
        return (
            dbc.Row(
                dbc.Label(
                    self.lang["CCI_BULLISH_VALUE"],
                    width=45,
                    style={"margin-top": "10px"},
                    class_name="center",
                ),
                class_name="center",
            ),
        )

    @property
    def trend_indicators_input1(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checklist(
                            self.dropdown_menu_item.indicators_filter,
                            id="checklist",
                            class_name="vertical-items-container",
                            input_class_name="btn-check",
                            label_class_name="btn btn-primary",
                            label_checked_class_name="active",
                            inline=True,
                        ),
                    ),
                ],
                class_name="center",
            ),
        )

    @property
    def trend_indicators_input2(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.DropdownMenu(
                            self.dropdown_menu_item.source_ohlc_items,
                            label=self.lang["SOURCE"],
                            id="source_crossover_column",
                        ),
                    ),
                ],
                class_name="center",
                style={"justify-content": "normal"},
            ),
        )

    @property
    def trend_indicators_input3(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id="indicator_macd_histogram_trend_value",
                            value=0,
                            type="number",
                        ),
                    ),
                ],
                class_name="center",
            ),
        )

    @property
    def trend_indicators_input4(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id="indicator_cci_trend_value",
                            value=0,
                            type="number",
                        ),
                    ),
                ],
                class_name="center",
            ),
        )

    @property
    def filter_components(self):
        return (
            dbc.Row(
                [
                    dbc.Col(self.trend_indicators_label1, width=6),
                    dbc.Col(self.trend_indicators_input1, width=6),
                ],
                class_name="center-row",
            ),
            dbc.Row(
                [
                    dbc.Col(self.trend_indicators_label2, width=6),
                    dbc.Col(
                        self.trend_indicators_input2,
                        style={"justify-content": "normal"},
                        width=6,
                    ),
                ],
                class_name="center-row",
            ),
            dbc.Row(
                [
                    dbc.Col(self.trend_indicators_label3, width=6),
                    dbc.Col(self.trend_indicators_input3, width=6),
                ],
                class_name="center-row",
            ),
            dbc.Row(
                [
                    dbc.Col(self.trend_indicators_label4, width=6),
                    dbc.Col(self.trend_indicators_input4, width=6),
                ],
                class_name="center-row",
            ),
        )

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
                    dbc.RadioItems(
                        result_types(self.lang),
                        id="result_types",
                        input_class_name="btn-check",
                        label_class_name="btn btn-primary",
                        label_checked_class_name="active",
                        inline=True,
                        value="Fixed",
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
            style={"justify-content": "center"},
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
                    class_name="center",
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
                dbc.Col(
                    dbc.Label(self.lang["RISK_FREE_RATE"]),
                    width=45,
                    style={"margin-top": "10px"},
                    class_name="center",
                ),
                dbc.Col(
                    dbc.Input(
                        id="risk_free_rate",
                        value=2.0,
                        type="number",
                        step=0.1,
                        min=0.0,
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
                    id="home_result_margin_type_col",
                ),
            ]
        )
