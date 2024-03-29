import pathlib
import dash
from dash import Input, Output, State, callback, html
import pandas as pd
import numpy as np

from controller.api import KlineAPI, KlineTimes

from model.utils import (
    interval_to_milliseconds,
    Statistics,
    SaveDataFrame,
    DataProcess,
)

from model.backtest import (
    Backtest,
    EmaParamsBacktest,
    IrbParamsBacktest,
    IndicatorsParamsBacktest,
    TrendParamsBacktest,
    ResultParamsBacktest,
    BacktestParams,
)

from view.dashboard.graph import GraphLayout
from view.dashboard.utils import get_data
from view.dashboard.pages.general.utils import content_parser, table_component


class RunBacktest:
    @callback(
        Output("backtest_result_margin_type_row", "class_name"),
        Input("result_percentage", "value"),
        State("backtest_result_types", "value"),
    )
    def show_margin_type(result_percentage, result_types):
        if len(result_percentage) == 1 and "Normal" in result_types:
            return "center"
        return "hidden"

    @callback(
        Output("backtest_results", "figure"),
        Output("backtest_text_output", "children"),
        Output("backtest_table_component", "children"),
        # Get Data
        Input("backtest_run_button", "n_clicks"),
        State("api_types", "value"),
        State("symbol", "value"),
        State("interval", "value"),
        State("custom-interval", "value"),
        # Custom Data
        State("custom_get_data-data", "contents"),
        State("custom_get_data-data", "filename"),
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
        #! If this options are enabled, the backtest will take a
        #! long time to run and will consume a lot of memory
        # # Trend Params - min values
        # State("backtest_min_indicator_macd_histogram_trend_value", "value"),
        # State("backtest_min_indicator_cci_trend_value", "value"),
        # # Trend Params - max values
        # State("backtest_max_indicator_macd_histogram_trend_value", "value"),
        # State("backtest_max_indicator_cci_trend_value", "value"),
        # Hardware Params
        State("hardware_types", "value"),
        State("backtest_cpu_cores_number", "value"),
        State("backtest_gpu_number", "value"),
        State("backtest_workers_number", "value"),
        State("backtest_result_types", "value"),
        State("result_percentage", "value"),
        State("initial_capital_value", "value"),
        State("qty_result_value", "value"),
        State("gain_result_value", "value"),
        State("loss_result_value", "value"),
        State("risk_free_rate", "value"),
        State("result_margin_type", "value"),
        State("plot_type", "value"),
    )
    def run_backtest(
        # Get Data
        backtest_run_button,
        api_type,
        symbol,
        interval,
        custom_interval,
        contents,
        filename,
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
        #! If this options are enabled, the backtest will take a
        #! long time to run and will consume a lot of memory
        # # Trend Params - min values
        # backtest_min_indicator_macd_histogram_trend_value,
        # backtest_min_indicator_cci_trend_value,
        # # Trend Params - max values
        # backtest_max_indicator_macd_histogram_trend_value,
        # backtest_max_indicator_cci_trend_value,
        # Hardware Params
        hardware_type,
        backtest_cpu_cores_number,
        backtest_gpu_number,
        backtest_workers_number,
        # Result Params
        backtest_result_types,
        result_percentage,
        initial_capital_value,
        qty_result_value,
        gain_result_value,
        loss_result_value,
        risk_free_rate,
        result_margin_type,
        result_type,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "backtest_run_button" in ctx.triggered[0]["prop_id"]:

            # Avoid errors when the symbol is in lowercase
            symbol = symbol.upper()

            if interval == "Custom"  and api_type != "custom":
                interval = custom_interval
                data_interval = KlineTimes(symbol, interval).get_max_interval
            else:
                data_interval = interval


            if result_percentage is None or result_percentage == []:
                result_percentage = False
            else:
                result_percentage = result_percentage[0]

            if api_type in ("coin_margined", "mark_price"):
                if symbol.endswith("USD"):
                    data_symbol = f"{symbol}_PERP"
                else:
                    data_symbol = f"{symbol}"
            else:
                data_symbol = symbol

            data_path = pathlib.Path("model", "data")
            data_name = f"{data_symbol}_{data_interval}_{api_type}"
            data_file = f"{data_name}.parquet"
            dataframe_path = data_path.joinpath(data_file)

            if dataframe_path.is_file() and api_type != "custom":
                data_frame = pd.read_parquet(dataframe_path)
                kline_api = KlineAPI(data_symbol, interval, api_type)
                data_frames = kline_api.update_data()
                updated_dataframe = data_frames[0]

                if kline_api.custom_interval:
                    data_frame = data_frames[1]
                else:
                    data_frame = updated_dataframe

                data_frame = data_frame.drop_duplicates()
                SaveDataFrame(updated_dataframe).to_parquet(f"{data_name}")

            else:
                if api_type == "custom":
                    data_frame = content_parser(contents, filename)
                    if data_frame:
                        data_frame = data_frame.drop_duplicates()
                else:
                    data_frame = (
                        get_data(data_symbol, data_interval, api_type)
                        .drop_duplicates()
                    )

                    SaveDataFrame(data_frame).to_parquet(f"{data_name}")


            ema_range = range(min_backtest_ema_length, max_backtest_ema_length + 1)
            lowest_low_range = range(backtest_min_lowestlow, backtest_max_lowestlow + 1)
            payoff_range = range(backtest_min_payoff, backtest_max_payoff + 1)

            wick_percentage_range = np.arange(
                backtest_min_wick_percentage,
                backtest_max_wick_percentage + 0.01,
                0.01,
            )

            # In Real App this variables setting should be dynamic
            backtest_min_indicator_macd_histogram_trend_value = 0
            backtest_max_indicator_macd_histogram_trend_value = 0
            backtest_min_indicator_cci_trend_value = 0
            backtest_max_indicator_cci_trend_value = 0

            macd_histogram_trend_value_range = range(
                backtest_min_indicator_macd_histogram_trend_value,
                backtest_max_indicator_macd_histogram_trend_value + 1,
            )

            cci_trend_value_range = range(
                backtest_min_indicator_cci_trend_value,
                backtest_max_indicator_cci_trend_value + 1,
            )

            backtest_params = BacktestParams(
                ema_params=EmaParamsBacktest(
                    source_column=["open", "high", "low", "close"],
                    length=list(ema_range),
                ),
                irb_params=IrbParamsBacktest(
                    lowestlow=list(lowest_low_range),
                    payoff=list(payoff_range),
                    ticksize=[0.1],
                    wick_percentage=list(np.round(wick_percentage_range, 2)),
                ),
                indicators_params=IndicatorsParamsBacktest(
                    ema_column=["open", "high", "low", "close"],
                    macd_histogram_trend_value=[list(macd_histogram_trend_value_range)],
                    cci_trend_value=list(cci_trend_value_range),
                ),
                trend_params=TrendParamsBacktest(
                    ema=[True],
                    macd=[False],
                    cci=[False],
                ),
                result_params=ResultParamsBacktest(
                    capital=[initial_capital_value],
                    percent=[result_percentage],
                    gain=[gain_result_value],
                    loss=[loss_result_value],
                    method=backtest_result_types,
                    coin_margined=[result_margin_type],
                ),
            )
            backtest = Backtest(data_frame, hardware_type)
            data_frame = backtest.param_grid_backtest(
                column="Capital",
                params=backtest_params,
                n_jobs=backtest_cpu_cores_number,
                n_gpu=backtest_gpu_number,
                n_workers_per_gpu=backtest_workers_number,
            )

            interval_to_hours = interval_to_milliseconds(interval) / 1000 / 60 / 60
            interval_to_year = 24 / interval_to_hours * 365
            risk_free_adjusted = risk_free_rate / interval_to_year

            if result_type:
                data_frame = DataProcess(
                    data_frame,
                    backtest_params.result_params.capital,
                ).best_positive_results

            if data_frame.shape[1] <= 50:
                range_max = data_frame.shape[1]
            else:
                range_max = 50

            stacked_dataframe = []
            for value in range(0, range_max):
                column_name = data_frame.columns[value]
                stats_df = data_frame[[column_name]].diff()
                stats_df = stats_df[stats_df[column_name] != 0][column_name]

                stats_df = Statistics(
                    stats_df,
                    risk_free_rate=risk_free_adjusted,
                    is_percent=True,
                ).calculate_all_statistics()

                if backtest_result_types == "Fixed" and not result_percentage:
                    stats_df = stats_df.drop("Sortino_Ratio", axis=1)

                stats_df["Rank"] = value + 1

                stats_df = stats_df.reindex(
                    columns=["Rank"]
                    + list(stats_df.columns[:-1])
                )

                stats_df.iloc[1:, 0] = None

                stacked_dataframe.append(stats_df)

            stacked_dataframe = pd.concat(stacked_dataframe)

            table = table_component(stacked_dataframe, "backtest_results-table")

            graph_layout = GraphLayout(
                data_frame,
                data_symbol,
                interval,
                api_type,
            )

            fig = graph_layout.grouped_lines()
            text_output = (
                f"Best Result: {data_frame.iloc[-1,0]}",
                html.Br(),
                f"Number of Trials: {backtest_params.total_combinations}",
            )

        return fig, text_output, table
