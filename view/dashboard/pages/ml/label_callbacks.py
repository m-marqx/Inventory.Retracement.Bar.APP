import os
import dash
from dash import Output, Input, State, callback

from model.machine_learning.utils import DataHandler
from model.machine_learning.features_creator import FeaturesCreator
from .utils import get_model_feat_params

class MLLabelCallbacks:
    @callback(
        Output("preset_options", "options"),
        Input("preset", "value"),
    )
    def update_preset_options(preset_selected):
        environ_var = os.getenv(preset_selected)

        if environ_var:
            return environ_var.replace(" ","").split(",") + ["All"]
        return ['None']