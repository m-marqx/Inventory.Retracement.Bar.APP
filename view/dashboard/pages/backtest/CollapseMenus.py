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
        ).component
