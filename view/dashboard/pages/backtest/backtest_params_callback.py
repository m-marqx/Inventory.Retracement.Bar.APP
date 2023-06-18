import dash
from dash import Input, Output, State, callback

class BacktestParams:
    @callback(
        Output("hardware_params_collapse", "is_open"),
        Output("hardware_params_icon", "className"),
        Input("hardware_params_button", "n_clicks"),
        State("hardware_params_collapse", "is_open"),
    )
    def toggle_hardware_params_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"

        return True, "fa fa-chevron-up ml-2"

    @callback(
        Output("result_configs_collapse", "is_open"),
        Output("result_configs_icon", "className"),
        Input("result_configs_button", "n_clicks"),
        State("result_configs_collapse", "is_open"),
    )
    def toggle_result_configs_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"

        return True, "fa fa-chevron-up ml-2"
