import os
import json

import dash
import tradingview_indicators as ta
import pandas as pd
from dash import Output, Input, State, callback
import time
import ccxt
import dash_ag_grid as dag

from controller.api.ccxt_api import CcxtAPI


from model.machine_learning.utils import DataHandler
from model.machine_learning.feature_params import FeaturesParams, FeaturesParamsComplete
from model.machine_learning.features_creator import FeaturesCreator

from .utils import get_model_feat_params
from .graph import GraphLayout


class RunModel:
    @callback(
        Output("progress_bar", "value", True),
        Input("run_model", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_progress_bar(run_button):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "run_model" in ctx.triggered[0]["prop_id"]:
            time.sleep(0.786)
            return "5"

    @callback(
        Output("model_text_output", "children"),
        Output("new_signal_output", "children"),
        Output("ml_results", "figure"),
        Output("text_model_spinner", "spinner_class_name"),
        Output("progress_bar", "value"),
        inputs=[
        Input("run_model", "n_clicks"),
        State("preset_options", "value"),
        Input("preset", "value"),
        ],
        background=True,
        cancel=Input("cancel_model", "n_clicks"),
        running=[
            (Output("run_model", "disabled"), True, False),
            (Output("cancel_model", "disabled"), False, True),
            (
                Output("progress_bar", "style"),
                {"visibility": "visible", "margin-top": "10px"},
                {"visibility": "hidden", "margin-top": "10px"},
            ),
        ],
        progress=Output("progress_bar", "value"),
        prevent_initial_call=True,
    )
    def get_model_predict(set_progress, run_button, preset_selected, preset_input):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "run_model" in ctx.triggered[0]["prop_id"]:
            set_progress("35")

            updated_dataset = pd.read_parquet("model/data/dataset_updated.parquet")
            BTCUSD = DataHandler(updated_dataset).calculate_targets()
            capi = CcxtAPI(symbol="BTC/USDT", interval="1d", exchange=ccxt.binance())
            updated_dataset = capi.update_klines(updated_dataset).drop(columns="volume")
            BTCUSD = DataHandler(updated_dataset.copy()).calculate_targets()
            BTCUSD["RSI79"] = ta.RSI(BTCUSD["high"].pct_change(), 79)
            return_series = BTCUSD["Return"]

            set_progress("25")

            split_params = dict(
                target_input=BTCUSD["Target_1_bin"],
                column="temp_indicator",
                log_values=True,
                threshold=0.50
            )

            split_params_H = dict(
                target_input=BTCUSD["Target_1_bin"],
                column="temp_indicator",
                log_values=True,
                threshold=0.52
            )

            split_params_L = dict(
                target_input=BTCUSD["Target_1_bin"],
                column="temp_indicator",
                log_values=True,
                threshold=0.48,
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


            try:
                model_params = json.loads(os.getenv('model_params'))
            except KeyError:
                model_params = {
                    "objective" : "binary:logistic",
                    "random_state" : 33,
                }

            predict_params = dict(
                indicator="rolling_ratio",
                model_params=model_params,
                fee=0.13,
                train_end_index=1527,
                features=["RSI79_low"],
            )

            validation_date = "2020-04-11 00:00:00"

            model_predict = FeaturesCreator(
                BTCUSD,
                return_series,
                BTCUSD["RSI79"],
                complete_params,
                validation_date
            )

            set_progress("35")

            model_predict.calculate_features("RSI79", 1527)
            combination = os.getenv(preset_selected)

            set_progress("45")

            if preset_selected == "All":
                set_progress("55")
                name_list = os.getenv(preset_input).replace(" ","").split(",")

                feats_list = [os.getenv(name) for name in name_list]


                params_list = [
                    get_model_feat_params(combination)
                    for combination in feats_list
                ]


                set_progress("95")

                result = {
                    name:
                    model_predict.calculate_model_returns(param, **predict_params)[combination]
                    for combination, param, name in zip(feats_list, params_list, name_list)
                }

                predict = [result[name]["Predict"].rename(name) for name in name_list]
                predict = pd.concat([result[name]["Predict"].rename(name) for name in name_list], axis=1)

                liquid_returns = [result[name]["Liquid_Return"].rename(name) for name in name_list]
                liquid_returns_concat = pd.concat(liquid_returns, axis=1)

                fig = GraphLayout(
                    liquid_returns_concat,
                    "BTC/USDT",
                    "1D",
                    "spot",
                ).grouped_lines()

            elif combination:
                set_progress("95")

                param = get_model_feat_params(combination)

                result = (
                    model_predict
                    .calculate_model_returns(param, **predict_params)
                    [combination]
                )

                predict = result["Predict"].to_frame()

                graph_layout = GraphLayout(
                    result,
                    "BTC/USDT",
                    "1D",
                    "spot",
                )

                fig = graph_layout.plot_single_linechart("Liquid_Return")

            recommendation = predict.shift()
            recommendation = (
                recommendation
                .where(recommendation < 0, 'Long')
                .where(recommendation > 0, 'Short')
                .tail(5)
            )

            recommendation = recommendation.reset_index()

            recommendation_table = dag.AgGrid(
                    rowData=recommendation.to_dict("records"),
                    getRowId="params.data.date",
                    columnDefs=[{"field": i} for i in recommendation.columns],
                    defaultColDef={"resizable": True, "sortable": True, "filter": True},
                    columnSize="responsiveSizeToFit",
                    dashGridOptions={"pagination": False},
                    className="ag-theme-alpine-dark",
                    style={"overflow": "hidden", "height": "29vh"},
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
                recommendation_table,
                new_signal_table,
                fig,
                'spinner-loader_model',
                "0",
            )
