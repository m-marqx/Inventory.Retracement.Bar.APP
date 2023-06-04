import pathlib
import dash
from dash import Input, Output, State
import pandas as pd
import numpy as np
from controller.api.klines_api import KlineAPI
from model.strategy.params import (
    EmaParams,
    MACDParams,
    CCIParams,
    TrendParams,
    IrbParams,
    IndicatorsParams,
)

from model.backtest import Backtest, BacktestParams
from model.utils.utils import SaveDataFrame, DataProcess

from view.dashboard.layout import app
from view.dashboard.graph import GraphLayout
from view.dashboard.utils import (
    BuilderParams,
    get_data,
    builder,
)

from view.dashboard.pages.lang.lang_callbacks import LangCallbacks
from view.dashboard.pages.home.label_callbacks import LabelCallbacks
from view.dashboard.pages.home.strategy_callback import RunStrategy
from view.dashboard.pages.backtest.backtest_callback import RunBacktest
from view.dashboard.pages.backtest.backtest_params_callback import BacktestParams
from view.dashboard.pages.general.general_collapse_callbacks import GeneralCollapse
from view.dashboard.pages.general.nav_link_pages_callbacks import NavLinkPages

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
