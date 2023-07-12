import pathlib
import pandas as pd

import dash
from dash import Input, Output, State, callback

from binance.helpers import interval_to_milliseconds

from controller.api.klines_api import KlineAPI

from model.utils import Statistics
from model.utils.utils import SaveDataFrame
from model.strategy.params import (
    EmaParams,
    MACDParams,
    CCIParams,
    TrendParams,
    IrbParams,
    IndicatorsParams,
    ResultParams,
)

from view.dashboard.graph import GraphLayout
from view.dashboard.utils import (
    BuilderParams,
    get_data,
    builder,
)

from view.dashboard.pages.general.utils import content_parser, table_component


class RunStrategy:
    @callback(
        Output("results", "figure"),
        Output("text_output", "children"),
        Output("table_container", "children"),
        Input("run_button", "n_clicks"),
        State("api_types", "value"),
        State("symbol", "value"),
        State("interval", "label"),
        State("custom_get_data-data", "contents"),
        State("custom_get_data-data", "filename"),
        State("ema_source_column", "label"),
        State("ema_length", "value"),
        State("macd_source_column", "label"),
        State("macd_fast_length", "value"),
        State("macd_slow_length", "value"),
        State("macd_signal_length", "value"),
        State("cci_source_column", "label"),
        State("cci_length", "value"),
        State("cci_ma_type", "label"),
        State("cci_constant", "value"),
        State("irb_lowestlow", "value"),
        State("irb_payoff", "value"),
        State("irb_tick_size", "value"),
        State("irb_wick_percentage", "value"),
        State("source_crossover_column", "label"),
        State("indicator_macd_histogram_trend_value", "value"),
        State("indicator_cci_trend_value", "value"),
        State("checklist", "value"),
        State("result_types", "value"),
        State("result_percentage", "value"),
        State("initial_capital_value", "value"),
        State("qty_result_value", "value"),
        State("gain_result_value", "value"),
        State("loss_result_value", "value"),
        State("risk_free_rate", "value"),
        State("result_margin_type", "value"),
    )
    def run_strategy(
        run_button,
        api_type,
        symbol,
        interval,
        contents,
        filename,
        ema_source_column,
        ema_length,
        macd_source_column,
        macd_fast_length,
        macd_slow_length,
        macd_signal_length,
        cci_source_column,
        cci_length,
        cci_ma_type,
        cci_constant,
        irb_lowestlow,
        irb_payoff,
        irb_tick_size,
        irb_wick_percentage,
        source_crossover_column,
        indicator_macd_histogram_trend_value,
        indicator_cci_trend_value,
        checklist,
        result_types,
        result_percentage,
        initial_capital_value,
        qty_result_value,
        gain_result_value,
        loss_result_value,
        risk_free_rate,
        result_margin_type,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "run_button" in ctx.triggered[0]["prop_id"]:
            symbol = symbol.upper()  # Avoid errors when the symbol is in lowercase

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
            data_name = f"{data_symbol}_{interval}_{api_type}"
            data_file = f"{data_name}.parquet"
            dataframe_path = data_path.joinpath(data_file)

            if dataframe_path.is_file() and api_type != "custom":
                data_frame = pd.read_parquet(dataframe_path)
                kline_api = KlineAPI(data_symbol, interval, api_type)
                data_frame = kline_api.update_data()

            else:
                if api_type == "custom":
                    data_frame = content_parser(contents, filename)
                else:
                    data_frame = get_data(data_symbol, interval, api_type)

            data_frame.drop_duplicates(inplace=True)
            if api_type != "custom":
                SaveDataFrame(data_frame).to_parquet(f"{data_name}")

            ema_bool = False
            macd_bool = False
            cci_bool = False

            if checklist is not None:
                for bool_param in checklist:
                    if bool_param == "ema":
                        ema_bool = True
                    if bool_param == "macd":
                        macd_bool = True
                    if bool_param == "cci":
                        cci_bool = True

            builder_params = BuilderParams(
                ema_params=EmaParams(
                    source_column=ema_source_column,
                    length=ema_length,
                ),
                macd_params=MACDParams(
                    source_column=macd_source_column,
                    fast_length=macd_fast_length,
                    slow_length=macd_slow_length,
                    signal_length=macd_signal_length,
                ),
                cci_params=CCIParams(
                    source_column=cci_source_column,
                    length=cci_length,
                    ma_type=cci_ma_type,
                    constant=cci_constant,
                ),
                irb_params=IrbParams(
                    lowestlow=irb_lowestlow,
                    payoff=irb_payoff,
                    tick_size=irb_tick_size,
                    wick_percentage=irb_wick_percentage,
                ),
                indicator_params=IndicatorsParams(
                    ema_column=source_crossover_column,
                    macd_histogram_trend_value=indicator_macd_histogram_trend_value,
                    cci_trend_value=indicator_cci_trend_value,
                ),
                trend_params=TrendParams(
                    ema=ema_bool,
                    macd=macd_bool,
                    cci=cci_bool,
                ),
                result_params = ResultParams(
                    capital = initial_capital_value,
                    percent = result_percentage,
                    gain = gain_result_value,
                    loss = loss_result_value,
                    method = result_types,
                    qty = qty_result_value,
                    coin_margined = result_margin_type,
                )
            )

            data_frame = builder(data_frame, builder_params)

            stats_dataframe = (
                ((data_frame[["Capital"]] - 100_000) / 100_000)
                .diff()
                .query("Capital != 0")
                ["Capital"]
            )

            interval_to_hours = interval_to_milliseconds(interval) / 1000 / 60 / 60
            interval_to_year = 24 / interval_to_hours * 365
            risk_free_adjusted = risk_free_rate / interval_to_year

            stats_df = Statistics(
                dataframe=stats_dataframe,
                risk_free_rate=risk_free_adjusted,
                is_percent=True,
            ).calculate_all_statistics()

            if result_types == "Fixed" and not result_percentage:
                stats_df = stats_df.drop("Sortino_Ratio", axis=1)

            table = table_component(stats_df, "results-table")

            graph_layout = GraphLayout(
                data_frame,
                data_symbol,
                interval,
                api_type
            )

            fig = graph_layout.plot_single_linechart("Capital")
            text_output = f"Final Result = {data_frame.iloc[-1,-1]:.2f}"
            return fig, text_output, table
