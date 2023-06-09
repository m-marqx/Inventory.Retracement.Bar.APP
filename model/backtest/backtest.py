from itertools import product
import torch
import gpuparallel
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import pandas as pd

from model.strategy.params import (
    EmaParams,
    MACDParams,
    CCIParams,
    IrbParams,
    IndicatorsParams,
    TrendParams,
    ResultParams,
)

from model.strategy.strategy import BuilderStrategy
from model.strategy.indicators import BuilderSource
from model.backtest.backtest_params import BacktestParams

class Backtest:
    def __init__(self, dataframe: pd.DataFrame, hardware_type: str = "GPU"):
        self.dataframe = dataframe.copy()

        self.strategy_df = pd.DataFrame()
        self.parameters_list = []

        if torch.cuda.is_available() and hardware_type == "GPU":
            self.hardware = "GPU"
        else:
            self.hardware = "CPU"

    def strategy(
        self,
        ema_params: EmaParams,
        irb_params: IrbParams,
        indicators: IndicatorsParams,
        trend: TrendParams,
        result_params: ResultParams,
    ):
        self.parameters_list = [
            f"EMA: {ema_params} <br> "
            f"IRB: {irb_params} <br> "
            f"Indicators: {indicators} <br> "
            f"Filters: {trend}"
            f"Result: {result_params}"
        ]

        self.strategy_df = (
            BuilderSource(
                self.dataframe,
            )
            .set_EMA_params(ema_params)
            .set_ema()
            .execute()
        )
        self.strategy_df = (
            BuilderStrategy(
                self.strategy_df,
            )
            .set_trend_params(indicators, trend)
            .get_trend()
            .set_irb_params(irb_params)
            .get_irb_signals()
            .calculate_irb_signals()
            .set_result_params(result_params)
            .calculateResults()
            .execute()
        )
        return self.strategy_df, self.parameters_list

    def run_backtest(self, ema_params, irb_params, indicators, trend, result_params, **kwargs):
        return self.strategy(
            ema_params,
            irb_params,
            indicators,
            trend,
            result_params,
        )

    def param_grid_backtest(
        self,
        column="Capital",
        n_jobs=-1,
        params: BacktestParams = BacktestParams(),
        n_gpu=1,
        n_workers_per_gpu=4,
    ):
        param_grid = {
            'ema_params': ParameterGrid(dict(params.ema_params)),
            'irb_params': ParameterGrid(dict(params.irb_params)),
            'indicators_params': ParameterGrid(dict(params.indicators_params)),
            'trend_params': ParameterGrid(dict(params.trend_params)),
            'result_params': ParameterGrid(dict(params.result_params)),
        }

        if self.hardware == "GPU":
            parallelizer = gpuparallel.GPUParallel(n_gpu, n_workers_per_gpu, progressbar=False)
            delayed_strategy = gpuparallel.delayed(self.run_backtest)
        else:
            parallelizer = Parallel(n_jobs)
            delayed_strategy = delayed(self.run_backtest)

        results = parallelizer(
            delayed_strategy(
                EmaParams(**dict(params[0])),
                IrbParams(**dict(params[1])),
                IndicatorsParams(**dict(params[2])),
                TrendParams(**dict(params[3])),
                ResultParams(**dict(params[4])),
            ) for params in product(*param_grid.values())
        )

        df_result = {}

        for params, arr in zip(product(*param_grid.values()), results):
            df_result[arr[1][-1]] = arr[0][column]

        return pd.DataFrame(df_result)
