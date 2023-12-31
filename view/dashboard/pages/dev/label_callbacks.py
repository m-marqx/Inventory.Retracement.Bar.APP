import os
import dash
from dash import Output, Input, State, callback

class DevMLLabelCallbacks:
    @callback(
        Output("dev_preset_options", "options"),
        Input("dev_preset", "value"),
    )
    def update_preset_options(preset_selected):
        environ_var = os.getenv(preset_selected)

        if environ_var:
            return environ_var.replace(" ","").split(",") + ["All"]
        return ['None']