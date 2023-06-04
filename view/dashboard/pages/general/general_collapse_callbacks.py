import dash
from dash import Output, Input, State, callback

class GeneralCollapse:
    @callback(
        Output("strategy_params_collapse", "is_open"),
        Output("strategy_params_icon", "className"),
        Input("strategy_params_button", "n_clicks"),
        State("strategy_params_collapse", "is_open"),
    )
    def toggle_strategy_params_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"
        else:
            return True, "fa fa-chevron-up ml-2"


    @callback(
        Output("get_data_collapse", "is_open"),
        Output("get_data_icon", "className"),
        Input("get_data_button", "n_clicks"),
        State("get_data_collapse", "is_open"),
    )
    def toggle_get_data_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down"
        else:
            return True, "fa fa-chevron-up"


    @callback(
        Output("indicator_params_collapse", "is_open"),
        Output("indicator_params_icon", "className"),
        Input("indicator_params_button", "n_clicks"),
        State("indicator_params_collapse", "is_open"),
    )
    def toggle_indicator_params_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"
        else:
            return True, "fa fa-chevron-up ml-2"


    @callback(
        Output("trend_params_collapse", "is_open"),
        Output("trend_params_icon", "className"),
        Input("trend_params_button", "n_clicks"),
        State("trend_params_collapse", "is_open"),
    )
    def toggle_strategy_params_collapse(n_clicks, is_open):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if is_open:
            return False, "fa fa-chevron-down ml-2"
        else:
            return True, "fa fa-chevron-up ml-2"
