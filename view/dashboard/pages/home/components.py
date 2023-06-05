import dash_bootstrap_components as dbc

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
    def macd_bullish_value_components(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["MACD_BULLISH_VALUE"]),
                    width=45,
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="indicator_macd_histogram_trend_value",
                        value=0,
                        type="number",
                    ),
                    width=45,
                ),
            ]
        )

    @property
    def cci_bullish_value_components(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["CCI_BULLISH_VALUE"]),
                    width=45,
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.Input(
                        id="indicator_cci_trend_value",
                        value=0,
                        type="number",
                    ),
                    width=45,
                ),
            ]
        )

    @property
    def macd_cci_components(self):
        return dbc.Row(
            [
                dbc.Col(dbc.CardGroup([self.macd_bullish_value_components])),
                dbc.Col(dbc.CardGroup([self.cci_bullish_value_components])),
            ]
        )

    @property
    def filter_components(self):
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["CROSSOVER_PRICE_SOURCE"]),
                    width=45,
                    style={"margin-top": "10px"},
                ),
                dbc.Col(
                    dbc.DropdownMenu(
                        children=self.dropdown_menu_item.source_ohlc_items,
                        label=self.lang["SOURCE"],
                        id="source_crossover_column",
                    ),
                    width=45,
                ),
                self.macd_cci_components,
                dbc.Col(
                    dbc.Label(
                        self.lang["ACTIVATE_INDICATOR"],
                        style={
                            "font-weight": "bold",
                            "font-size": "20px",
                            "margin-top": "10px",
                        },
                    ),
                    width=45,
                ),
                dbc.Col(
                    dbc.Checklist(
                        self.dropdown_menu_item.indicators_filter,
                        id="checklist",
                        input_class_name="btn-check",
                        label_class_name="btn btn-primary",
                        label_checked_class_name="active",
                        inline=True,
                    )
                ),
            ]
        )
