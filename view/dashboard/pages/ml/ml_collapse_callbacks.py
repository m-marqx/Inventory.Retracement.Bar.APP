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

