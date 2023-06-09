import pathlib
import dash
from dash import Input, Output, State, callback
import pandas as pd
import numpy as np
from controller.api.klines_api import KlineAPI


from model.backtest import Backtest, BacktestParams
from model.utils.utils import SaveDataFrame, DataProcess

from view.dashboard.graph import GraphLayout
from view.dashboard.utils import (
    get_data,
)

class RunBacktest:
    @callback(
        Output("backtest_results", "figure"),
        Output("backtest_text_output", "children"),
        # Get Data
        Input("backtest_run_button", "n_clicks"),
        State("api_types", "value"),
        State("symbol", "value"),
        State("interval", "label"),
        # Indicators Params
        State("min_backtest_ema_length", "value"),
        State("max_backtest_ema_length", "value"),
        # Strategy Params - min values
        State("backtest_min_lowestlow", "value"),
        State("backtest_min_payoff", "value"),
        State("backtest_min_wick_percentage", "value"),
        # Strategy Params - max values
        State("backtest_max_lowestlow", "value"),
        State("backtest_max_payoff", "value"),
        State("backtest_max_wick_percentage", "value"),
        # Trend Params - min values
        State("backtest_min_indicator_macd_histogram_trend_value", "value"),
        State("backtest_min_indicator_cci_trend_value", "value"),
        # Trend Params - max values
        State("backtest_max_indicator_macd_histogram_trend_value", "value"),
        State("backtest_max_indicator_cci_trend_value", "value"),
        # Hardware Params
        State("hardware_types", "value"),
        State("backtest_cpu_cores_number", "value"),
        State("backtest_gpu_number", "value"),
        State("backtest_workers_number", "value"),
    )
    def run_backtest(
        # Get Data
        backtest_run_button,
        api_type,
        symbol,
        interval,
        # Indicators Params
        min_backtest_ema_length,
        max_backtest_ema_length,
        # Strategy Params - min values
        backtest_min_lowestlow,
        backtest_min_payoff,
        backtest_min_wick_percentage,
        # Strategy Params - max values
        backtest_max_lowestlow,
        backtest_max_payoff,
        backtest_max_wick_percentage,
        # Trend Params - min values
        backtest_min_indicator_macd_histogram_trend_value,
        backtest_min_indicator_cci_trend_value,
        # Trend Params - max values
        backtest_max_indicator_macd_histogram_trend_value,
        backtest_max_indicator_cci_trend_value,
        # Hardware Params
        hardware_type,
        backtest_cpu_cores_number,
        backtest_gpu_number,
        backtest_workers_number,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "backtest_run_button" in ctx.triggered[0]["prop_id"]:
            symbol = symbol.upper()  # Avoid errors when the symbol is in lowercase

            if api_type in ("coin_margined", "mark_price"):
                if symbol.endswith("USD"):
                    data_symbol = f"{symbol}_PERP"
                else:
                    data_symbol = f"{symbol}"
            else:
                data_symbol = symbol

            data_path = pathlib.Path("model", "data")
            data_name = f"{data_symbol}_{interval}_{api_type}"
            data_file = f"{data_name}.parquet"
            dataframe_path = data_path.joinpath(data_file)

            if dataframe_path.is_file():
                data_frame = pd.read_parquet(dataframe_path)
                kline_api = KlineAPI(data_symbol, interval, api_type)
                data_frame = kline_api.update_data()

            else:
                data_frame = get_data(data_symbol, interval, api_type)

            # For some reason, the data in Deploy is aways duplicated.
            data_frame.drop_duplicates(inplace=True)
            SaveDataFrame(data_frame).to_parquet(f"{data_name}")

            ema_range = range(min_backtest_ema_length, max_backtest_ema_length + 1)
            lowest_low_range = range(backtest_min_lowestlow, backtest_max_lowestlow + 1)
            payoff_range = range(backtest_min_payoff, backtest_max_payoff + 1)

            wick_percentage_range = np.arange(
                backtest_min_wick_percentage,
                backtest_max_wick_percentage + 0.01,
                0.01,
            )

            macd_histogram_trend_value_range = range(
                backtest_min_indicator_macd_histogram_trend_value,
                backtest_max_indicator_macd_histogram_trend_value + 1,
            )

            cci_trend_value_range = range(
                backtest_min_indicator_cci_trend_value,
                backtest_max_indicator_cci_trend_value + 1,
            )

            backtest_params = BacktestParams(
                ema_params={
                    "length": list(ema_range),
                    "source_column": ["open", "high", "low", "close"],
                },
                irb_params={
                    "lowestlow": list(lowest_low_range),
                    "payoff": list(payoff_range),
                    "ticksize": [0.1],
                    "wick_percentage": np.round(wick_percentage_range, 2).tolist(),
                },
                indicators_params={
                    "ema_column": ["open", "high", "low", "close"],
                    "macd_histogram_trend_value": list(macd_histogram_trend_value_range),
                    "cci_trend_value": list(cci_trend_value_range),
                },
                trend_params={
                    "ema": [True],
                    "macd": [False],
                    "cci": [False],
                },
            )
            backtest = Backtest(data_frame, hardware_type)
            data_frame = backtest.param_grid_backtest(
                params=backtest_params,
                n_jobs=backtest_cpu_cores_number,
                n_gpu=backtest_gpu_number,
                n_workers_per_gpu=backtest_workers_number,
            )

            data_frame = DataProcess(data_frame).best_positive_results

            graph_layout = GraphLayout(
                data_frame,
                data_symbol,
                interval,
                api_type,
            )

            fig = graph_layout.grouped_lines()
            text_output = f"Best Result: {data_frame.iloc[-1,0]}"

        return fig, text_output
