from itertools import product
import torch
import gpuparallel
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import pandas as pd

from model.strategy.params import (
    EmaParams,
    IrbParams,
    IndicatorsParams,
    TrendParams,
    ResultParams,
)

from model.backtest.backtest_params import BacktestParams

from view.dashboard.utils import (
    BuilderParams,
    builder,
)


class Backtest:
    """
    A class for performing backtesting on a given dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe for backtesting.
    hardware_type : str, optional
        The type of hardware to use for backtesting (default is "GPU").

    Attributes
    ----------
    dataframe : pd.DataFrame
        The input dataframe for backtesting.
    strategy_df : pd.DataFrame
        The dataframe storing the backtesting strategy results.
    parameters_list : list
        A list of strategy parameters.

    Methods
    -------
    strategy(ema_parameters, irb_parameters, indicator_parameters, trend_parameters, result_parameters)
        Run the backtesting strategy with the specified parameters.
    run_backtest(ema_params, irb_params, indicators, trend, result_params, **kwargs)
        Run a single backtest with the specified parameters.
    param_grid_backtest(column, n_jobs, params, n_gpu, n_workers_per_gpu)
        Run backtests with a parameter grid and return the results.
    """

    def __init__(self, dataframe: pd.DataFrame, hardware_type: str = "GPU"):
        """
        Initialize a Backtest instance.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input dataframe for backtesting.
        hardware_type : str, optional
            The type of hardware to use for backtesting (default is "GPU").
        """
        self.dataframe = dataframe.copy()
        self.strategy_df = pd.DataFrame()
        self.parameters_list = []

        if torch.cuda.is_available() and hardware_type == "GPU":
            self.hardware = "GPU"
        else:
            self.hardware = "CPU"

    def strategy(
        self,
        ema_parameters: EmaParams,
        irb_parameters: IrbParams,
        indicator_parameters: IndicatorsParams,
        trend_parameters: TrendParams,
        result_parameters: ResultParams,
    ):
        """
        Run the backtesting strategy with the specified parameters.

        Parameters
        ----------
        ema_parameters : EmaParams
            The parameters for the EMA strategy.
        irb_parameters : IrbParams
            The parameters for the IRB strategy.
        indicator_parameters : IndicatorsParams
            The parameters for the indicator strategy.
        trend_parameters : TrendParams
            The parameters for the trend strategy.
        result_parameters : ResultParams
            The parameters for the result strategy.

        Returns
        -------
        tuple
            A tuple containing the strategy dataframe and the parameters
            list.
        """
        self.parameters_list = [
            f"EMA: {ema_parameters} <br> "
            f"IRB: {irb_parameters} <br> "
            f"Indicators: {indicator_parameters} <br> "
            f"Filters: {trend_parameters} <br> "
            f"Result: {result_parameters} <br> "
        ]

        strategy_params = BuilderParams(
            ema_params=ema_parameters,
            irb_params=irb_parameters,
            indicators_params=indicator_parameters,
            trend_params=trend_parameters,
            result_params=result_parameters,
        )

        self.strategy_df = builder(self.dataframe, strategy_params)

        return self.strategy_df, self.parameters_list

    def run_backtest(
        self,
        ema_params,
        irb_params,
        indicators,
        trend,
        result_params,
        **kwargs
    ):
        """
        Run a single backtest with the specified parameters.

        Parameters
        ----------
        ema_params : dict
            The parameters for the EMA strategy.
        irb_params : dict
            The parameters for the IRB strategy.
        indicators : dict
            The parameters for the indicator strategy.
        trend : dict
            The parameters for the trend strategy.
        result_params : dict
            The parameters for the result strategy.

        Returns
        -------
        tuple
            A tuple containing the strategy dataframe and the parameters
            list.
        """
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
        """
        Run backtests with a parameter grid and return the results.

        Parameters
        ----------
        column : str, optional
            The column to use for the results (default is "Capital").
        n_jobs : int, optional
            The number of parallel jobs to run (-1 means using all
            available processors, default is -1).
        params : BacktestParams, optional
            The parameters for the backtest
            (default is BacktestParams()).
        n_gpu : int, optional
            The number of GPUs to use for parallelization
            (default is 1).
        n_workers_per_gpu : int, optional
            The number of workers per GPU
            (default is 4).

        Returns
        -------
        pd.DataFrame
            A dataframe containing the backtest results.
        """
        param_grid = {
            'ema_params': ParameterGrid(dict(params.ema_params)),
            'irb_params': ParameterGrid(dict(params.irb_params)),
            'indicators_params': ParameterGrid(dict(params.indicators_params)),
            'trend_params': ParameterGrid(dict(params.trend_params)),
            'result_params': ParameterGrid(dict(params.result_params)),
        }

        if self.hardware == "GPU":

            parallelizer = gpuparallel.GPUParallel(
                n_gpu,
                n_workers_per_gpu,
                progressbar=False
            )

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

        results_dict = {}

        for params, arr in zip(product(*param_grid.values()), results):
            results_dict[arr[1][-1]] = arr[0][column]

        return pd.DataFrame(results_dict)
