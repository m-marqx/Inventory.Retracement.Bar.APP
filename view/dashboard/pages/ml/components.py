import dash_bootstrap_components as dbc
from dash import dcc


class MLComponents:
    """A class representing the components of a main page in a Dash
    application.

    Parameters
    ----------
    lang : dict
        A dictionary containing language translations.

    Attributes
    ----------
    lang : dict
        A dictionary containing language translations.

    """

    def __init__(self, lang):
        self.lang = lang

    @property
    def presets_getter(self):
        """Component for the preset selection variable.

        Returns
        -------
        dbc.Col
            A Col component containing the preset selection variable.

        """
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label("PRESET"),
                    width=45,
                    style={"margin-top": "1vh"},
                    class_name="center",
                ),
            dbc.Col(
                dbc.Input(
                    id="preset",
                    value="presets",
                    placeholder="PRESETS in environment variables",
                    type="text",
                ),
                width=45
            )
            ],
            style={"justify-content": "center"},
        )


    @property
    def preset_setter(self):
        """Components for the preset selection.

        Returns
        -------
        dbc.Row
            A Row component containing the preset selection components.

        """
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["COMBINATION"]),
                    width=45,
                    style={"margin-top": "1vh"},
                    class_name="center",
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="preset_options",
                        placeholder=self.lang["SELECT_PLACEHOLDER"],
                    ),
                    width=45,
                ),
            ],
            style={"justify-content": "center"},
        )

    @property
    def preset_settings(self):
        return dbc.Col(
            [
                self.presets_getter,
                self.preset_setter,
            ],
            class_name="d-grid gap-2 col-6 mx-auto w-100 menu-collapse_container",
        )

    @property
    def collapse_preset_settings(self):
        return MenuCollapse(
            lang=self.lang,
            label="USE_PREBUILD_MODELS",
            component=self.preset_settings,
            id_prefix="preset_configs",
        ).simple_collapse

