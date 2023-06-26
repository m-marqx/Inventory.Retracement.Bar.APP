import dash_bootstrap_components as dbc
from view.dashboard.pages.general.utils import MenuCollapse
from view.dashboard.pages.general.components import GeneralComponents
from view.dashboard.pages.home.components import MainPageComponents
from view.dashboard.pages.backtest.components import BacktestComponents
from view.dashboard.pages.backtest.results_components import ResultsComponents


class LayoutMenuCollapse:
    def __init__(self, lang, page):
        self.lang = lang
        self.general_components = GeneralComponents(lang)
        if page == "home":
            self.page_component_base = MainPageComponents(lang)
            self.page_component_result = MainPageComponents(lang)
        else:
            self.backtest_components = BacktestComponents(lang)
            self.page_component_base = BacktestComponents(lang)
            self.page_component_result = ResultsComponents(lang)

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
                    self.page_component_base.indicators_parameters_col1,
                    width=6,
                ),
                dbc.Col(
                    self.page_component_base.indicators_parameters_col2,
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
                    self.page_component_base.irb_parameters_col1,
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                    },
                    width=6,
                ),
                dbc.Col(
                    self.page_component_base.irb_parameters_col2,
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
            component=dbc.Col(self.page_component_base.filter_components),
            id_prefix="trend_params",
        ).simple_collapse

    @property
    def result_parameters_component(self):
        result_configs_component = MenuCollapse(
            lang=self.lang,
            label="MODIFY_RESULT_CONFIGS_BUTTON",
            component=self.page_component_result.result_type,
            id_prefix="result_configs",
        ).simple_collapse

        return MenuCollapse(
            lang=self.lang,
            label="MODIFY_RESULT_PARAMETERS_BUTTON",
            component=self.page_component_result.result_components,
            id_prefix="result_params",
        ).collapse_with_inside_collapse(result_configs_component)

