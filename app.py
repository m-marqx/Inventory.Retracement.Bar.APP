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

from view.dashboard.pages.ml.ml_callback import RunModel
from view.dashboard.pages.ml.label_callbacks import MLLabelCallbacks

server = app.server

if __name__ == "__main__":
    LabelCallbacks()
    LangCallbacks()
    RunStrategy()
    RunBacktest()
    MLLabelCallbacks()
    RunModel()
    BacktestParams()
    GeneralCollapse()
    NavLinkPages()
    app.run(debug=False)
