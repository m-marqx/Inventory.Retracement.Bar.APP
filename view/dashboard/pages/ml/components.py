import dash_bootstrap_components as dbc
from dash import dcc

from view.dashboard.pages import (
    MenuCollapse,
    DropdownMenu,
)

from view.dashboard.pages.ml.utils import scorings, eval_metric

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
                            id='rsi_length-input',
                            type='number',
                            value=89,
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
                            id='first_length-input',
                            type='number',
                            value=144
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
                            id='second_length-input',
                            type='number',
                            value=233
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

    @property
    def feat_params_components(self):
        thresholds = [dbc.Row(
            [
                dbc.Col(
                    dbc.Label(self.lang["FEATURES_THRESHOLD"]),
                    width=45,
                    class_name="center",
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Col(
                            dbc.Label(self.lang["HIGH"]),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="high_threshold",
                            value=0.52,
                            min=0,
                            max=1,
                            step=0.01,
                            placeholder="High threshold",
                            type="number",
                        )],
                        style={"width": "30%", "margin-right": "auto"},
                    ),
                    dbc.Col([
                        dbc.Col(
                            dbc.Label(self.lang["INTERMEDIATE"]),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="split_threshold",
                            value=0.50,
                            min=0,
                            max=1,
                            step=0.01,
                            placeholder="Intermediate threshold",
                            type="number",
                        )],
                        style={"width": "30%", "margin-right": "auto"},
                    ),
                    dbc.Col([
                        dbc.Col(
                            dbc.Label(self.lang["LOW"]),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="low_threshold",
                            value=0.48,
                            min=0,
                            max=1,
                            step=0.01,
                            placeholder="Low threshold",
                            type="number",
                        )],
                        style={"width": "30%"},
                    )
                ],
                style={"justify-content": "center"},
                ),
            ]
        )]

    @property
    def model_param_component(self):
        model_params_row = [
                dbc.Col(
                    dbc.Label(self.lang["MODEL_PARAMS"]),
                    width=45,
                    class_name="center",
                    style={"margin-top": "2vh"},
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Col(
                            dbc.Label("n_estimators"),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="n_estimators",
                            min=0,
                            step=1,
                            type="number",
                            value=333,
                        )],
                        style={"width": "30%", "margin-right": "auto"},
                    ),
                    dbc.Col([
                        dbc.Col(
                            dbc.Label("Max Depth"),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="max_depth",
                            min=0,
                            step=1,
                            type="number",
                            value=11,
                        )],
                        style={"width": "30%", "margin-right": "auto"},
                    ),
                    dbc.Col([
                        dbc.Col(
                            dbc.Label("Gamma"),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="gamma",
                            min=0,
                            step=1,
                            type="number",
                            value=7,
                        )],
                        style={"width": "30%"},
                    )
                ],
                style={"justify-content": "center"},
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Col(
                            dbc.Label("Subsample"),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="subsample",
                            min=0,
                            max=1,
                            step=0.01,
                            type="number",
                            value=0.69,
                        )],
                        style={"width": "30%", "margin-right": "auto"},
                    ),
                    dbc.Col([
                        dbc.Col(
                            dbc.Label("Learning Rate"),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="learning_rate",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.69,
                            type="number",
                        )],
                        style={"width": "30%", "margin-right": "auto"},
                    ),
                    dbc.Col([
                        dbc.Col(
                            dbc.Label("Colsample Bytree"),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="colsample_bytree",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.07,
                            type="number",
                        )],
                        style={"width": "30%"},
                    )
                ],
                style={"justify-content": "center"},
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Col(
                            dbc.Label("Random State"),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dbc.Input(
                            id="random_state",
                            value=69,
                            min=0,
                            step=1,
                            type="number",
                        )],
                        style={"width": "30%", "margin-right": "auto"},
                    ),
                    dbc.Col([
                        dbc.Col(
                            dbc.Label("Eval Metric"),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dcc.Dropdown(
                            eval_metric,
                            id="eval_metric",
                            placeholder=self.lang['SELECT_PLACEHOLDER'],
                            value="logloss",
                        )],
                        style={"width": "30%", "margin-right": "auto"},
                    ),
                    dbc.Col([
                        dbc.Col(
                            dbc.Label(self.lang["SCORING_METRIC"]),
                            width=45,
                            style={"margin-top": "1vh"},
                            class_name="center",
                        ),
                        dcc.Dropdown(
                            scorings,
                            id="scorings",
                            value="accuracy",
                            placeholder=self.lang['SELECT_PLACEHOLDER'],
                        )],
                        style={"width": "30%", "margin-right": "auto"},
                    )],
                style={"justify-content": "center"},
                ),
            ]

        button = dbc.Button(
            self.lang["GENERATE_PARAMS"],
            id="generate_params",
            color="primary",
            className="mr-1",
            style={"margin-top": "2.5vh", "width": "40%", "margin-right": "auto", "margin-left": "auto"},
            outline=True,
        )

        return dbc.Col(model_params_row + [button])

    @property
    def model_params_settings(self):
        return MenuCollapse(
            lang=self.lang,
            label="MODIFY_PARAMS",
            component=dbc.Col([self.feat_params_components, self.model_param_component]),
            id_prefix="model_params",
            style={"margin-top": "1vh"},
        ).simple_collapse

    @property
    def dataset_split_settings(self):
        return MenuCollapse(
            lang=self.lang,
            label="MODIFY_DATASET_SPLITS",
            component=self.dataset_split_component,
            id_prefix="dataset_splits",
            style={"margin-top": "1vh"},
        ).simple_collapse

    @property
    def dataset_split_component(self):
        return (
            dbc.Row([
                dbc.Col([
                    dbc.Col(
                        dbc.Label(self.lang["TRAIN_TEST_PERIOD"]),
                        width=45,
                        class_name="center",
                    ),
                    dcc.DatePickerRange(
                        id="train_test_date",
                        min_date_allowed="2012-01-02",
                        className="center",
                        start_date="2012-01-02",
                        end_date="2020-04-12",
                        start_date_id="start_train_test_date",
                        end_date_id="end_train_test_date",
                        display_format="DD/MM/YYYY",
                        show_outside_days=True,
                    ),
                ]),
                dbc.Col([
                    dbc.Col(
                        dbc.Label(self.lang["TRAIN_TEST_RATIO"]),
                        width=45,
                        class_name="center",
                    ),
                    dbc.Input(
                        id="TRAIN_TEST_RATIO",
                        step=0.01,
                        value=0.5,
                        max=1,
                        min=0,
                    ),
                ]),
            ]),
            dbc.Col([
                dbc.Checklist(
                    [{"label":"Use Custom Validation Date", "value":True}],
                    id="validation_checklist",
                    class_name="is_custom_validation_check",
                    input_class_name="btn-check",
                    label_class_name="btn btn-outline-primary",
                    label_checked_class_name="active",
                    style={"margin-top": "2vh", "margin-bottom": "1vh", "width": "27.36vh", "padding-left": "0px"},
                ),
                dcc.DatePickerSingle(
                    id="validation_date",
                    min_date_allowed="2012-01-02",
                    className="hidden",
                    display_format="DD/MM/YYYY"
                )], style={"margin-left": "auto", "margin-right": "auto", "justify-items": "center", "display": "grid"},
            )
        )
