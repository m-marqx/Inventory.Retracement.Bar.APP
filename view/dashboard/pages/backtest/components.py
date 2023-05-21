import dash_bootstrap_components as dbc
from dash import html, dcc


class BacktestComponents:
    def __init__(self, lang):
        self.lang = lang

    @property
    def indicators_first_col(self):
        return (
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Label(
                            self.lang["MIN_EMA_LENGTH"],
                            html_for="ema_length",
                            width=45,
                        ),
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="min_backtest_ema_length",
                            value=0,
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
                            html_for="ema_length",
                            width=45,
                        ),
                        style={"margin-top": "10px"},
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="max_backtest_ema_length",
                            value=100,
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
    def macd_bullish_value_components(self):
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
    def macd_cci_components(self):
        return (
            dbc.Row(
                [
                    dbc.Col(dbc.CardGroup([self.macd_bullish_value_components])),
                    dbc.Col(dbc.CardGroup([self.cci_bullish_value_components])),
                ]
            )
        )

    @property
    def filter_components(self):
        return (
            dbc.Row(
                [
                    self.macd_cci_components,
                    dbc.Col(
                        html.Label(
                            self.lang["ACTIVATE_INDICATOR"],
                            style={
                                "font-weight": "bold",
                                "font-size": "20px",
                                "margin-top": "10px",
                            },
                        ),
                        width=45,
                    ),
                ]
            )
        )

    @property
    def filter_components_col1(self):
        return dbc.CardGroup([self.filter_components])