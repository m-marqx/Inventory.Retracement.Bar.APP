import dash
from dash import Input, Output, State, callback

class DevMLCollapseCallbacks:
    @callback(
        Output("dev_preset_configs_collapse", "is_open"),
        Output("dev_preset_configs_icon", "className"),
        Input("dev_preset_configs_button", "n_clicks"),
        State("dev_preset_configs_collapse", "is_open"),
    )
    def toggle_dev_preset_configs_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"
        return True, "fa fa-chevron-up ml-2"
