import dash
from dash import Input, Output, State, callback

class MLCollapseCallbacks:
    @callback(
        Output("preset_configs_collapse", "is_open"),
        Output("preset_configs_icon", "className"),
        Input("preset_configs_button", "n_clicks"),
        State("preset_configs_collapse", "is_open"),
    )
    def toggle_preset_configs_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"

        return True, "fa fa-chevron-up ml-2"

    @callback(
        Output("indicators_models_collapse", "is_open"),
        Output("indicators_models_icon", "className"),
        Input("indicators_models_button", "n_clicks"),
        State("indicators_models_collapse", "is_open"),
    )
    def toggle_indicators_configs_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"

        return True, "fa fa-chevron-up ml-2"

    @callback(
        Output("model_params_collapse", "is_open"),
        Output("model_params_icon", "className"),
        Input("model_params_button", "n_clicks"),
        State("model_params_collapse", "is_open"),
    )
    def toggle_model_params_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"

        return True, "fa fa-chevron-up ml-2"

    @callback(
        Output("dataset_splits_collapse", "is_open"),
        Output("dataset_splits_icon", "className"),
        Input("dataset_splits_button", "n_clicks"),
        State("dataset_splits_collapse", "is_open"),
    )
    def toggle_dataset_splits_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"

        return True, "fa fa-chevron-up ml-2"
