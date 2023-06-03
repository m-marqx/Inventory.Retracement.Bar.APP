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
    ):
        self.parameters_list = [
            f"EMA: {ema_params} <br> "
            f"IRB: {irb_params} <br> "
            f"Indicators: {indicators} <br> "
            f"Filters: {trend}"
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
            .calculateResults()
            .execute()
        )
        return self.strategy_df, self.parameters_list

    def run_backtest(self, ema_params, irb_params, indicators, trend, **kwargs):
        return self.strategy(
            ema_params,
            irb_params,
            indicators,
            trend,
        )

    def param_grid_backtest(
        self,
        column="Cumulative_Result",
        n_jobs=-1,
        params: BacktestParams = BacktestParams(),
        n_gpu=1,
        n_workers_per_gpu=4,
    ):
        param_grid = {
            'ema_params': ParameterGrid(params.ema_params.dict()),
            'irb_params': ParameterGrid(params.irb_params.dict()),
            'indicators_params': ParameterGrid(params.indicators_params.dict()),
            'trend_params': ParameterGrid(params.trend_params.dict()),
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
            ) for params in product(*param_grid.values())
        )

        df_result = {}

        for params, arr in zip(product(*param_grid.values()), results):
            df_result[arr[1][-1]] = arr[0][column]

        return pd.DataFrame(df_result)

    def best_positive_results(self, backtest_params: BacktestParams):
        backtest = Backtest(self.dataframe)
        backtest_df = backtest.param_grid_backtest(params=backtest_params)
        transposed_df = backtest_df.T
        transposed_df_last_column = transposed_df.iloc[:, [-1]]

        filtered_df = transposed_df_last_column[transposed_df_last_column > 0]
        filtered_df.dropna(inplace=True)

        filtered_df_sorted = filtered_df.sort_values(
            by=str(filtered_df.columns[-1]),
            ascending=False,
        ).index

        return transposed_df.loc[filtered_df_sorted].T
