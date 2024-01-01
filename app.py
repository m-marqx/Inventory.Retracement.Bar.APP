from view.dashboard.layout import app
from view.dashboard.pages import (
    LangCallbacks,
    LabelCallbacks,
    RunStrategy,
    RunBacktest,
    BacktestParams,
    GeneralCollapse,
    NavLinkPages,
)

from view.dashboard.pages.dev.ml_callback import DevRunModel
from view.dashboard.pages.dev.label_callbacks import DevMLLabelCallbacks
from view.dashboard.pages.dev.ml_collapse_callbacks import DevMLCollapseCallbacks

from view.dashboard.pages.ml.label_callbacks import MLLabelCallbacks
from view.dashboard.pages.ml.ml_collapse_callbacks import MLCollapseCallbacks
from view.dashboard.pages.ml.model_ml_callback import ModelMLCallback
from view.dashboard.pages.ml.inputs_callback import ModelInputsCallbacks
from view.dashboard.pages.ml.model_params_callback import ModelParamsCallback

server = app.server

if __name__ == "__main__":
    #General
    LangCallbacks()
    NavLinkPages()
    GeneralCollapse()

    #Home
    RunStrategy()
    LabelCallbacks()

    #Backtest
    RunBacktest()
    BacktestParams()

    #ML
    ModelMLCallback()
    MLLabelCallbacks()
    MLCollapseCallbacks()
    ModelInputsCallbacks()
    ModelParamsCallback()

    #DEV
    DevRunModel()
    DevMLLabelCallbacks()
    DevMLCollapseCallbacks()
    app.run(debug=True)
