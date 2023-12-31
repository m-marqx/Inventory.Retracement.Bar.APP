import dash
from dash import Output, Input, State, callback

class ModelInputsCallbacks:
    @callback(
        Output("features_selection_dropdown", "options"),
        Input("indicators_dropdown", "value"),
    )
    def update_features_selection_dropdown(indicators_dropdown):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        feat_selection = []

        if "RSI" in indicators_dropdown:

            feat_selection.extend([
                "RSI_low",
                "RSI_split",
                "RSI_high",
            ])
        if "rolling ratio" in indicators_dropdown:

            feat_selection.extend([
                "rolling_ratio_low",
                "rolling_ratio_split",
                "rolling_ratio_high",
            ])

        return feat_selection

    @callback(
        Output("rolling_ratio_dropdown", "options"),
        Input("indicators_dropdown", "value"),
    )
    def update_rolling_ratio_dropdown(indicators):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "RSI" in indicators:
            return ["open", "high", "low", "close", "RSI"]
        return ["open", "high", "low", "close"]

    @callback(
        Output("rsi_settings", "class_name"),
        Output("rolling_settings", "class_name"),
        Input("indicators_dropdown", "value"),
    )
    def update_rolling_ratio_dropdown(indicators):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        is_rsi_selected = False
        is_rolling_ratio_selected = False

        if "RSI" in indicators:
            is_rsi_selected = True
        if "rolling ratio" in indicators:
            is_rolling_ratio_selected = True

        if is_rsi_selected and not is_rolling_ratio_selected:
            return "row", "hidden"
        elif is_rolling_ratio_selected and not is_rsi_selected:
            return "hidden", "row"
        elif is_rsi_selected and is_rolling_ratio_selected:
            return "row", "row"
        return "hidden", "hidden"

    @callback(
        Output("validation_date", "className"),
        Output("validation_date", "date"),
        Input("validation_checklist", "value"),
        State("train_test_date", "end_date"),
    )
    def update_validation_date(is_custom_validation_date, last_date):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if is_custom_validation_date:
            return "center", last_date
        return "hidden", None
