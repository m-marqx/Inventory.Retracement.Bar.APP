import dash
import tradingview_indicators as ta
import numpy as np
import pandas as pd
from dash import Output, Input, State, callback
import ccxt
import dash_ag_grid as dag
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

from controller.api.ccxt_api import CcxtAPI

from model.machine_learning.utils import DataHandler
from model.machine_learning.feature_params import FeaturesParams, FeaturesParamsComplete
from model.machine_learning.features_creator import FeaturesCreator

from .graph import GraphLayout

class ModelMLCallback:
    @callback(
        Output("model_text_output", "children", True),
        Output("signal_output", "children", True),
        Output("new_signal_output", "children", True),
        Output("ml_results", "figure", True),
        Output("progress_bar", "children", True),
        Output("train_test_date", "start_date", True),
        inputs=[
            Input("run_model", "n_clicks"),

            #Data
            State("exchange_dropdown", "value"),
            State("interval_dropdown", "value"),
            State("symbol", "value"),

            # Indicators 26 - 54
            State("indicators_dropdown", "value"),
            State("rsi_model_dropdown", "value"),
            State("rsi_length-input", "value"),
            State("rolling_ratio_dropdown", "value"),
            State("first_length-input", "value"),
            State("second_length-input", "value"),

            # Features
            State("high_threshold", "value"),
            State("split_threshold", "value"),
            State("low_threshold", "value"),
            State("features_selection_dropdown", "value"),

            # Model Params
            State("n_estimators", "value"),
            State("max_depth", "value"),
            State("gamma", "value"),
            State("subsample", "value"),
            State("learning_rate", "value"),
            State("colsample_bytree", "value"),
            State("random_state", "value"),
            State("eval_metric", "value"),

            # Data Split
            State("train_test_date", "start_date"),
            State("train_test_date", "end_date"),
            State("TRAIN_TEST_RATIO", "value"),
            # State("validation_date", "date"),
        ],
        background=True,
        cancel=Input("cancel_model", "n_clicks"),
        running=[
            (Output("run_model", "disabled", True), True, False),
            (Output("cancel_model", "disabled", True), False, True),
            (Output("progress_bar", "className", True), "progress-info", "hidden"),
            (Output("signal_output", "className", True), "hidden", ""),
            (Output("new_signal_output", "className", True), "hidden", ""),
        ],
        progress=Output("progress_bar", "children"),
        prevent_initial_call=True,
    )
    def get_model_predicts(
        set_progress,
        run_button,

        #data
        exchange,
        interval,
        symbol,

        # Indicators 70-98
        indicators_selected,
        rsi_source,
        rsi_length,
        rolling_ratio_source,
        rolling_ratio_first_length,
        rolling_ratio_second_length,

        # Features
        high_threshold,
        split_threshold,
        low_threshold,
        features_selected,

        # Model Params
        n_estimators,
        max_depth,
        gamma,
        subsample,
        learning_rate,
        colsample_bytree,
        random_state,
        eval_metric,

        # Data Split
        test_train_start_date,
        test_train_end_date,
        TRAIN_TEST_RATIO,
        # validation_date,
    ):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "run_model" in ctx.triggered[0]["prop_id"]:

            set_progress("Loading data...")

            try:
                exchange = getattr(ccxt, exchange)()
                capi = CcxtAPI(symbol=symbol, interval=interval, exchange=exchange, verbose=True)
                exchange_klines = capi.get_all_klines().to_OHLCV().data_frame
            except Exception as e:
                return (
                    "API ERROR",
                    [],
                    [],
                    None,
                    "",
                    test_train_start_date,
                )

            set_progress("Creating model...")

            BTCUSD2 = DataHandler(exchange_klines).calculate_targets().loc[test_train_start_date:]
            return_series = BTCUSD2["Return"]

            train_start_date = pd.to_datetime(test_train_start_date)
            first_train_start_date = pd.to_datetime(return_series.index[0])

            train_diff_days = (train_start_date - first_train_start_date).total_seconds()

            first_train_day_index = (
                first_train_start_date if train_diff_days <= 0
                else train_start_date
            )

            print(first_train_day_index)

            split_params = dict(
                target_input=BTCUSD2["Target_1_bin"],
                column="temp_indicator",
                log_values=True,
                threshold=split_threshold
            )

            split_params_H = dict(
                target_input=BTCUSD2["Target_1_bin"],
                column="temp_indicator",
                log_values=True,
                threshold=high_threshold,
            )

            split_params_L = dict(
                target_input=BTCUSD2["Target_1_bin"],
                column="temp_indicator",
                log_values=True,
                threshold=low_threshold,
                higher_than_threshold=False,
            )

            split_params = FeaturesParams(**split_params)
            high_params = FeaturesParams(**split_params_H)
            low_params = FeaturesParams(**split_params_L)

            complete_params = FeaturesParamsComplete(
                split_features=split_params,
                high_features=high_params,
                low_features=low_params
            )

            model_params = {"objective" : "binary:logistic"}
            model_params["n_estimators"] = n_estimators
            model_params["max_depth"] = max_depth
            model_params["gamma"] = gamma
            model_params["subsample"] = subsample
            model_params["learning_rate"] = learning_rate
            model_params["colsample_bytree"] = colsample_bytree
            model_params["random_state"] = random_state
            model_params["eval_metric"] = eval_metric

            model_predict = FeaturesCreator(
                BTCUSD2, return_series,
                None,
                complete_params, test_train_end_date
            )


            train_end_index = int(
                BTCUSD2
                .loc[:test_train_end_date].shape[0]
                * TRAIN_TEST_RATIO
            )


            set_progress("Calculating Features...")

            is_rsi_indicator_selected = "RSI" in indicators_selected
            is_rolling_ratio_selected = "rolling ratio" in indicators_selected

            if is_rsi_indicator_selected:
                model_predict.data_frame["RSI"] = (
                    model_predict
                    .temp_indicator(rsi_length, "RSI", BTCUSD2[rsi_source].pct_change())
                )

                model_predict.calculate_features("RSI", train_end_index)

            if is_rolling_ratio_selected:
                ratio_value = [
                    rolling_ratio_first_length,
                    rolling_ratio_second_length,
                    "std"
                ]

                model_predict.data_frame["rolling_ratio"] = (
                    model_predict.temp_indicator(
                        ratio_value, "rolling_ratio",
                        model_predict.data_frame[rolling_ratio_source]
                    )
                )

                model_predict.calculate_features("rolling_ratio", train_end_index)

            set_progress("Calculating model returns...")

            if features_selected:
                results = model_predict.calculate_results(
                    features_selected,
                    model_params=model_params,
                    fee=0.13,
                    test_size=TRAIN_TEST_RATIO)

                predict = results["Predict"].to_frame()

                graph_layout = GraphLayout(
                    results,
                    "BTC/USDT",
                    "1D",
                    "spot",
                )

                fig = graph_layout.plot_single_linechart("Liquid_Return")
                fig.add_vline(
                    x=test_train_end_date,
                    line_width=1,
                    line_dash="dash",
                    line_color="#595959"
                )

                recommendation = predict.copy().shift()
                recommendation = (
                    recommendation
                    .where(recommendation < 0, "Long")
                    .where(recommendation > 0, "Short")
                    .tail(5)
                    .reset_index()
                )

                recommendation_table = dag.AgGrid(
                        rowData=recommendation.to_dict("records"),
                        getRowId="params.data.date",
                        columnDefs=[{"field": i} for i in recommendation.columns],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        columnSize="responsiveSizeToFit",
                        dashGridOptions={"pagination": False},
                        className="ag-theme-alpine-dark",
                        style={"overflow": "hidden", "height": "27vh", "margin-top": "1vh"},
                    )

                new_signal = pd.DataFrame({"Unconfirmed" : predict.iloc[-1]}).T

                new_signal = (
                    new_signal
                    .where(new_signal < 0, "Long")
                    .where(new_signal > 0, "Short")
                    .reset_index()
                )

                new_signal.columns = ["date"] + list(new_signal.columns[1:])

                new_signal_table = dag.AgGrid(
                        rowData=new_signal.to_dict("records"),
                        getRowId="params.data.date",
                        columnDefs=[{"field": i} for i in new_signal.columns],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        columnSize="responsiveSizeToFit",
                        dashGridOptions={"pagination": False},
                        className="ag-theme-alpine-dark",
                        style={"overflow": "hidden", "height": "10vh", "margin-top": "1vh"},
                    )

                return (
                    "",
                    recommendation_table,
                    new_signal_table,
                    fig,
                    "",
                    first_train_day_index,
                )
            return (
                "",
                [],
                [],
                None,
                "",
                first_train_day_index,
            )
