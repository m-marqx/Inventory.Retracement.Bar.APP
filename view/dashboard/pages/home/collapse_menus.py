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
