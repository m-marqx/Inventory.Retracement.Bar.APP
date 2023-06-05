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

server = app.server

if __name__ == "__main__":
    LabelCallbacks()
    LangCallbacks()
    RunStrategy()
    RunBacktest()
    BacktestParams()
    GeneralCollapse()
    NavLinkPages()
    app.run(debug=False)
