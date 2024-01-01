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

class ModelParamsCallback:
    @callback(
        # Model Params User Values
        Output("n_estimators", "value"),
        Output("max_depth", "value"),
        Output("gamma", "value"),
        Output("subsample", "value"),
        Output("learning_rate", "value"),
        Output("colsample_bytree", "value"),
        Output("train_test_date", "start_date", True),

        Output("n_estimators", "class_name"),
        Output("max_depth", "class_name"),
        Output("gamma", "class_name"),
        Output("subsample", "class_name"),
        Output("learning_rate", "class_name"),
        Output("colsample_bytree", "class_name"),

        # Model Params Inputs
        inputs=[
            Input("generate_params", "n_clicks"),
            #Data
            State("exchange_dropdown", "value"),
            State("interval_dropdown", "value"),
            State("symbol", "value"),

            State("random_state", "value"),
            State("eval_metric", "value"),
            State("scorings", "value"),

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

            # Data Split
            State("train_test_date", "start_date"),
            State("train_test_date", "end_date"),
            State("TRAIN_TEST_RATIO", "value"),

            #Dinamic Model Params Inputs
            State("n_estimators", "value"),
            State("max_depth", "value"),
            State("gamma", "value"),
            State("subsample", "value"),
            State("learning_rate", "value"),
            State("colsample_bytree", "value"),
        ],
        background=True,
        running=[
            (Output("generate_params", "disabled"), True, False),
            (Output("generate_spinner", "spinner_class_name"), "spinner-loader", "hidden")
        ],
        prevent_initial_call=True,

    )
    def model_random_search(
        generate_button,

        # Data
        exchange,
        interval,
        symbol,

        # Model Params User Values
        random_state,
        eval_metric,
        scorings,

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

        # Data Split
        test_train_start_date,
        test_train_end_date,
        TRAIN_TEST_RATIO,

        #Dinamic Model Params Inputs
        n_estimators,
        max_depth,
        gamma,
        subsample,
        learning_rate,
        colsample_bytree,
    ):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "generate_params" in ctx.triggered[0]["prop_id"]:
            exchange = getattr(ccxt, exchange)()
            capi = CcxtAPI(symbol=symbol, interval=interval, exchange=exchange)
            exchange_klines = capi.get_all_klines().to_OHLCV().data_frame

            BTCUSD2 = DataHandler(exchange_klines).calculate_targets().loc[test_train_start_date:]
            return_series = BTCUSD2["Return"]

            train_start_date = pd.to_datetime(test_train_start_date)
            first_train_start_date = pd.to_datetime(return_series.index[0])

            train_diff_days = (train_start_date - first_train_start_date).total_seconds()

            first_train_day_index = (
                first_train_start_date if train_diff_days <= 0
                else train_start_date
            )

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

            if features_selected:
                target = model_predict.data_frame["Target_1_bin"].dropna()
                features = model_predict.data_frame[features_selected].reindex(target.index)

                param_distributions = dict(
                    n_estimators = range(50, 1001, 50),
                    max_depth = range(3, 101, 1),
                    gamma = range(3, 101, 1),
                    subsample = np.arange(0.1, 1.01, 0.05).round(2),
                    learning_rate = np.arange(0.01, 1.01, 0.05).round(2),
                    colsample_bytree = np.arange(0.1, 1.01, 0.05).round(2),
                    eval_metric = [eval_metric]
                )

                model = xgb.XGBClassifier()

                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_distributions,
                    n_iter=100,
                    cv=4,
                    random_state=random_state,
                    n_jobs=-1,
                    scoring=scorings,
                )

                # Fit the RandomizedSearchCV object to the data
                random_search.fit(features, target)

                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "gamma": gamma,
                    "subsample": subsample,
                    "learning_rate": learning_rate,
                    "colsample_bytree": colsample_bytree,
                }

                # Get the best parameters
                best_params = random_search.best_params_

                new_model_params = {
                    key: "form-control" if model_params[key] == best_params[key]
                    else "focused-form-control" for key in model_params
                }

                return (
                    best_params["n_estimators"],
                    best_params["max_depth"],
                    best_params["gamma"],
                    best_params["subsample"],
                    best_params["learning_rate"],
                    best_params["colsample_bytree"],
                    first_train_day_index,

                    new_model_params["n_estimators"],
                    new_model_params["max_depth"],
                    new_model_params["gamma"],
                    new_model_params["subsample"],
                    new_model_params["learning_rate"],
                    new_model_params["colsample_bytree"],
                )
            return (
                "",
                "",
                "",
                "",
                "",
                "",
                first_train_day_index,
            )
