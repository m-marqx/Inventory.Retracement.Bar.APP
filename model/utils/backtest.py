import pandas as pd
import concurrent.futures

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

    def ema_backtest(self, start=1, end=100, column="Cumulative_Result"):
        df_result = {}
        results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            for col in ["open", "high", "low", "close"]:
                for value in range(start, end + 1, 1):
                    ema_params = EmaParams(length=value, source_column=col)
                    futures.append(
                        executor.submit(
                            self.run_backtest,
                            ema_params,
                            IrbParams(),
                            IndicatorsParams(),
                            TrendParams(ema=True),
                        )
                    )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                results_values = result[0][column]
                results_params = result[1][-1]
                df_result[results_params] = results_values

        df_result = pd.DataFrame(df_result)
        return df_result

    def wick_backtest(
        self,
        start=0,
        end=100,
        ema_length=20,
        column="Cumulative_Result",
    ):
        df_result = {}

        columns = ["open", "high", "low", "close"]
        for col in columns:
            for value in range(start, end + 1, 1):
                value = value / 100
                ema_params = EmaParams(length=ema_length, source_column=col)
                params = IrbParams(wick_percentage=value)
                results = self.strategy(
                    ema_params,
                    params,
                    IndicatorsParams(),
                    TrendParams(ema=True),
                )
                results_values = results[0][column]
                results_params = results[1][-1]
                df_result[results_params] = results_values

        df_result = pd.DataFrame(df_result)
        return df_result
