import numpy as np
from pydantic import BaseModel


class EmaParamsBacktest(BaseModel):
    """
    Parameters for the EMA strategy in backtesting.

    Parameters
    ----------
    length : list[int], optional
        The list of lengths for EMA calculation
        (default is range(1, 101)).
    source_column : list[str], optional
        The list of source columns for EMA calculation
        (default is ["open", "high", "low", "close"]).
    """
    length: list[int] = range(1, 101)
    source_column: list[str] = ["open", "high", "low", "close"]


class IrbParamsBacktest(BaseModel):
    """
    Parameters for the IRB strategy in backtesting.

    Parameters
    ----------
    lowestlow : list[int], optional
        The list of lowest low values for IRB calculation
        (default is range(1, 2)).
    payoff : list[int], optional
        The list of payoff values for IRB calculation
        (default is range(2, 3)).
    ticksize : list[float], optional
        The list of tick size values for IRB calculation
        (default is np.arange(0.1, 0.2)).
    wick_percentage : list[float], optional
        The list of wick percentage values for IRB calculation
        (default is np.round(np.arange(0.01, 1.01, 0.01), 2)).
    """
    lowestlow: list[int] = range(1, 2)
    payoff: list[int] = range(2, 3)
    ticksize: list[float] = np.arange(0.1, 0.2)
    wick_percentage: list[float] = np.round(np.arange(0.01, 1.01, 0.01), 2)


class IndicatorsParamsBacktest(BaseModel):
    """
    Parameters for the indicator strategy in backtesting.

    Parameters
    ----------
    ema_column : list[str], optional
        The list of source columns for EMA calculation in indicator
        strategy (default is ["open", "high", "low", "close"]).
    macd_histogram_trend_value : list[int], optional
        The list of MACD histogram trend values in indicator strategy
        (default is range(0, 1)).
    cci_trend_value : list[int], optional
        The list of CCI trend values in indicator strategy
        (default is range(0, 1)).
    """
    ema_column: list[str] = ["open", "high", "low", "close"]
    macd_histogram_trend_value: list[int] = range(0, 1)
    cci_trend_value: list[int] = range(0, 1)


class TrendParamsBacktest(BaseModel):
    """
    Parameters for the trend strategy in backtesting.

    Parameters
    ----------
    ema : list[bool], optional
        The list of boolean values indicating whether to use EMA in
        trend strategy (default is [True]).
    macd : list[bool], optional
        The list of boolean values indicating whether to use MACD in
        trend strategy (default is [False]).
    cci : list[bool], optional
        The list of boolean values indicating whether to use CCI in
        trend strategy (default is [False]).
    """
    ema: list[bool] = [True]
    macd: list[bool] = [False]
    cci: list[bool] = [False]


class ResultParamsBacktest(BaseModel):
    """
    Parameters for the result strategy in backtesting.

    Parameters
    ----------
    capital : list[float], optional
        The list of initial capital values (default is [100000]).
    percent : list[bool], optional
        The list of boolean values indicating whether to use percentage
        values (default is [True]).
    gain : list[float], optional
        The list of gain values (default is [2]).
    loss : list[float], optional
        The list of loss values (default is [-1]).
    method : list[str], optional
        The list of method values (default is ["Fixed"]).
    qty : list[float], optional
        The list of quantity values (default is [1]).
    coin_margined : list[bool], optional
        The list of boolean values indicating whether the coin is
        margined (default is [True]).
    """
    capital: list[float] = [100_000]
    percent: list[bool] = [True]
    gain: list[float] = [2]
    loss: list[float] = [-1]
    method: list[str] = ["Fixed"]
    qty: list[float] = [1]
    coin_margined: list[bool] = [True]


class BacktestParams(BaseModel):
    """
    Parameters for backtesting.

    Parameters
    ----------
    ema_params : EmaParamsBacktest, optional
        The parameters for the EMA strategy in backtesting
        (default is EmaParamsBacktest()).
    irb_params : IrbParamsBacktest, optional
        The parameters for the IRB strategy in backtesting
        (default is IrbParamsBacktest()).
    indicators_params : IndicatorsParamsBacktest, optional
        The parameters for the indicator strategy in backtesting
        (default is IndicatorsParamsBacktest()).
    trend_params : TrendParamsBacktest, optional
        The parameters for the trend strategy in backtesting
        (default is TrendParamsBacktest()).
    result_params : ResultParamsBacktest, optional
        The parameters for the result strategy in backtesting
        (default is ResultParamsBacktest()).

    Attributes
    ----------
    total_combinations : int
        The total number of combinations of all parameter values.
    """
    ema_params: EmaParamsBacktest = EmaParamsBacktest()
    irb_params: IrbParamsBacktest = IrbParamsBacktest()
    indicators_params: IndicatorsParamsBacktest = IndicatorsParamsBacktest()
    trend_params: TrendParamsBacktest = TrendParamsBacktest()
    result_params: ResultParamsBacktest = ResultParamsBacktest()

    @property
    def total_combinations(self):
        """
        Calculate the total number of combinations of all parameter
        values.

        Returns
        -------
        int
            The total number of combinations.
        """
        num_combinations = (
            len(self.ema_params.length)
            * len(self.ema_params.source_column)
            * len(self.irb_params.lowestlow)
            * len(self.irb_params.payoff)
            * len(self.irb_params.ticksize)
            * len(self.irb_params.wick_percentage)
            * len(self.indicators_params.ema_column)
            * len(self.indicators_params.macd_histogram_trend_value)
            * len(self.indicators_params.cci_trend_value)
            * len(self.trend_params.ema)
            * len(self.trend_params.macd)
            * len(self.trend_params.cci)
            * len(self.result_params.capital)
            * len(self.result_params.percent)
            * len(self.result_params.gain)
            * len(self.result_params.loss)
            * len(self.result_params.method)
            * len(self.result_params.qty)
            * len(self.result_params.coin_margined)
        )
        return num_combinations
