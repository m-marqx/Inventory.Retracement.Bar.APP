import dash_bootstrap_components as dbc
from view.dashboard.pages.general.utils import MenuCollapse
from view.dashboard.pages.general.components import GeneralComponents
from .components import MainPageComponents

class MainPageMenuCollapse:
    def __init__(self, lang):
        self.lang = lang
        self.general_components = GeneralComponents(lang)
        self.main_page_components = MainPageComponents(lang)

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
                    self.main_page_components.indicators_parameters_col1,
                    width=6,
                ),
                dbc.Col(
                    self.main_page_components.indicators_parameters_col2,
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
                    self.main_page_components.irb_parameters_col1,
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                    },
                    width=6,
                ),
                dbc.Col(
                    self.main_page_components.irb_parameters_col2,
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
