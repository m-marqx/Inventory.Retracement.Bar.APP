from itertools import product
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import pandas as pd

from model.strategy.params.indicators_params import (
    EmaParams,
    MACDParams,
    CCIParams,
)

from model.strategy.params.strategy_params import (
    IrbParams,
    IndicatorsParams,
    TrendParams,
)

from model.strategy.strategy import BuilderStrategy
from model.strategy.indicators import BuilderSource
from model.utils.backtest_params import BacktestParams

class Backtest:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()
        self.data_frame = pd.DataFrame()

    def strategy(
        self,
        ema_params: EmaParams,
        irb_params: IrbParams,
        indicators: IndicatorsParams,
        trend: TrendParams,
    ):
        self.parameters_list = [
            f"EMA: {ema_params} <br> "
            f"IRB: {irb_params} <br> "
            f"Indicators: {indicators} <br> "
            f"Filters: {trend}"
        ]

        self.data_frame = (
            BuilderSource(
                self.dataframe,
            )
            .set_EMA_params(ema_params)
            .set_ema()
            .execute()
        )
        self.data_frame = (
            BuilderStrategy(
                self.data_frame,
            )
            .set_trend_params(indicators, trend)
            .get_trend()
            .set_irb_params(irb_params)
            .get_irb_signals()
            .calculate_irb_signals()
            .calculateResults()
            .execute()
        )
        return self.data_frame, self.parameters_list

    def run_backtest(self, ema_params, irb_params, indicators, trend):
        backtest_df = self.dataframe.copy()
        backtest = Backtest(backtest_df)
        return backtest.strategy(
            ema_params,
            irb_params,
            indicators,
            trend,
        )

    def ema_backtest(
        self,
        column="Cumulative_Result",
        n_jobs=-1,
        params: BacktestParams = BacktestParams(),
    ):
        df_result = {}
        param_grid = {
            'ema_params': ParameterGrid(params().ema_params),
            'irb_params': [IrbParams()],
            'indicators_params': [IndicatorsParams()],
            'trend_params': [TrendParams(ema=True)]
        }

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.run_backtest)(
                EmaParams(**dict(params[0])),
                IrbParams(**dict(params[1])),
                IndicatorsParams(**dict(params[2])),
                TrendParams(**dict(params[3])),
            ) for params in product(*param_grid.values())
        )

        for params, arr in zip(product(*param_grid.values()), results):
            df_result[arr[1][-1]] = arr[0][column]

        return pd.DataFrame(df_result)

    def wick_backtest(
        self,
        column="Cumulative_Result",
        n_jobs=-1,
        params: BacktestParams = BacktestParams(),
    ):
        df_result = {}
        param_grid = {
            'ema_params': ParameterGrid(params.ema_params),
            'irb_params': ParameterGrid(params.irb_params),
            'indicators_params': [IndicatorsParams()],
            'trend_params': [TrendParams(ema=True)]
        }

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.run_backtest)(
                EmaParams(**dict(params[0])),
                IrbParams(**dict(params[1])),
                IndicatorsParams(**dict(params[2])),
                TrendParams(**dict(params[3])),
            ) for params in product(*param_grid.values())
        )

        for params, arr in zip(product(*param_grid.values()), results):
            df_result[arr[1][-1]] = arr[0][column]

        return pd.DataFrame(df_result)
