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

    @property
    def rsi_settings(self):
        return dbc.Row(
            [
                dbc.Col(
                    DropdownMenu(
                        lang=self.lang,
                        label="RSI",
                        options=["open", "high", "low", "close"],
                        id_prefix="rsi_model",
                        is_multi_options=False,
                    ).dropdown_components,
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Col(
                            dbc.Label(self.lang["LENGTH"]),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id='length-input',
                            type='number',
                        ),
                    ],
                    width=6,
                )
            ],
            id="rsi_settings",
            class_name="hidden"
        )

    @property
    def rolling_ratio_settings(self):
        return dbc.Row(
            [
                dbc.Col(
                    DropdownMenu(
                        lang=self.lang,
                        label="ROLLING_RATIO_SOURCE",
                        options=["open", "high", "low", "close", "RSI"],
                        id_prefix="rolling_ratio",
                    ).dropdown_components,
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Col(
                            dbc.Label(self.lang["FIRST_LENGTH"]),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id='length-input',
                            type='number',
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Col(
                            dbc.Label(self.lang["SECOND_LENGTH"]),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id='length-input',
                            type='number',
                        ),
                    ],
                    width=3,
                )
            ],
            id="rolling_settings",
            class_name="hidden"
        )

    @property
    def indicators_select(self):
        """Component for the dropdown menu selection.

        Returns
        -------
        dbc.Row
            A Row component containing the dropdown menu selection.

        """
        return dbc.Col(
            [
                DropdownMenu(
                    lang=self.lang,
                    label="INDICATORS",
                    options=["RSI", "rolling ratio"],
                    id_prefix="indicators",
                    is_multi_options=True,
                ).dropdown_components,
                self.rsi_settings,
                self.rolling_ratio_settings,
            ]
        )

    @property
    def indicators_settings(self):
        return MenuCollapse(
            lang=self.lang,
            label="MODIFY_INDICATORS",
            component=self.indicators_select,
            id_prefix="indicators_models",
            style={"margin-top": "1vh"},
        ).simple_collapse

