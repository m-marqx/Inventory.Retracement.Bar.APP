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
