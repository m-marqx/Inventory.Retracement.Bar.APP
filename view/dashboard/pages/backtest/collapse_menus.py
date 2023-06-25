import dash_bootstrap_components as dbc
from view.dashboard.pages.general.utils import MenuCollapse
from view.dashboard.pages.general.components import GeneralComponents
from .components import BacktestComponents
from .results_components import ResultsComponents


class BacktestMenuCollapse:
    def __init__(self, lang):
        self.lang = lang
        self.backtest_components = BacktestComponents(lang)
        self.general_components = GeneralComponents(lang)
        self.results_components = ResultsComponents(lang)

    @property
    def get_data_component(self):
        return MenuCollapse(
            lang=self.lang,
            label="GET_DATA_BUTTON",
            component=self.general_components.get_data_components,
            id_prefix="get_data",
            is_open=True,
        ).simple_collapse

    @property
    def parameters_component(self):
        indicators_parameters_component = dbc.Row(
            [
                dbc.Col(
                    self.backtest_components.indicators_parameters_col1,
                    width=6,
                ),
                dbc.Col(
                    self.backtest_components.indicators_parameters_col2,
                    width=6,
                ),
            ]
        )

        return MenuCollapse(
            lang=self.lang,
            label="MODIFY_INDICATORS_PARAMETERS_BUTTON",
            component=indicators_parameters_component,
            id_prefix="indicator_params",
        ).simple_collapse

    @property
    def strategy_component(self):
        irb_parameters = dbc.Row(
            [
                dbc.Col(
                    self.backtest_components.irb_parameters_col1,
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                    },
                    width=6,
                ),
                dbc.Col(
                    self.backtest_components.irb_parameters_col2,
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                    },
                    width=6,
                ),
            ]
        )

        return MenuCollapse(
            lang=self.lang,
            label="MODIFY_STRATEGY_PARAMETERS_BUTTON",
            component=irb_parameters,
            id_prefix="strategy_params",
        ).simple_collapse

    @property
    def trend_component(self):
        return MenuCollapse(
            lang=self.lang,
            label="MODIFY_TREND_PARAMETERS_BUTTON",
            component=dbc.Col(self.backtest_components.filter_components),
            id_prefix="trend_params",
        ).simple_collapse

    @property
    def result_parameters_component(self):
        result_configs_component = MenuCollapse(
            lang=self.lang,
            label="MODIFY_RESULT_CONFIGS_BUTTON",
            component=self.results_components.result_type,
            id_prefix="result_configs",
        ).simple_collapse

        return MenuCollapse(
            lang=self.lang,
            label="MODIFY_RESULT_PARAMETERS_BUTTON",
            component=self.results_components.result_components,
            id_prefix="result_params",
        ).collapse_with_inside_collapse(result_configs_component)

    @property
    def hardware_component(self):
        return MenuCollapse(
            lang=self.lang,
            label="MODIFY_HARDWARE_PARAMETERS_BUTTON",
            component=self.backtest_components.hardware_components,
            id_prefix="hardware_params",
        ).simple_collapse
