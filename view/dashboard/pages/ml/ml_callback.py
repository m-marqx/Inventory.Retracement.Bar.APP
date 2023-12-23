import os

import dash
from dash import Output, Input, State, callback
import tradingview_indicators as ta
import pandas as pd
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
        Output("model_text_output", "children"),
        Output("ml_results", "figure"),
        Input("run_model", "n_clicks"),
        State("preset_options", "value"),
        Input("preset", "value"),
    )
    def get_model_predict(run_button, preset_selected, preset_input):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        # print(ctx.triggered)
        if "run_model" in ctx.triggered[0]["prop_id"]:
            updated_dataset = pd.read_parquet('model/data/dataset_updated.parquet')

            last_update = updated_dataset.index[-1]

            BTCUSD = DataHandler(updated_dataset).calculate_targets()
            capi = CcxtAPI(symbol='BTC/USDT', interval='1d', exchange=ccxt.binance())
            updated_dataset = capi.update_klines(updated_dataset).drop(columns='volume')
            BTCUSD = DataHandler(updated_dataset.copy()).calculate_targets()
            BTCUSD['RSI79'] = ta.RSI(BTCUSD['high'].pct_change(), 79)
            return_series = BTCUSD['Return']

            split_params = dict(
                target_input=BTCUSD['Target_1_bin'],
                column='temp_indicator',
                log_values=True,
                threshold=0.50
            )

            split_params_H = dict(
                target_input=BTCUSD['Target_1_bin'],
                column='temp_indicator',
                log_values=True,
                threshold=0.52
            )

            split_params_L = dict(
                target_input=BTCUSD['Target_1_bin'],
                column='temp_indicator',
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

            model_params = {
                'objective' : "binary:logistic",
                'random_state' : 1635,
                'subsample': 0.6499999999999999,
                'n_estimators': 450,
                'max_depth': 16,
                'learning_rate': 0.66,
                'gamma': 5,
                'colsample_bytree': 0.1,
                'eval_metric' : 'auc'
            }

            predict_params = dict(
                indicator='rolling_ratio',
                model_params=model_params,
                fee=0.13,
                train_end_index=1527,
                features=['RSI79_low'],
            )

            validation_date = '2020-04-11 00:00:00'
            model_predict = FeaturesCreator(
                BTCUSD,
                return_series,
                BTCUSD['RSI79'],
                complete_params,
                validation_date
            )
            model_predict.calculate_features('RSI79', 1527)
            combination = os.getenv(preset_selected)
            if preset_selected == 'All':
                name_list = os.getenv(preset_input).replace(" ","").split(",")

                feats_list = [os.getenv(name) for name in name_list]

                params_list = [
                    get_model_feat_params(combination)
                    for combination in feats_list
                ]

                result = {
                    name:
                    model_predict.calculate_model_returns(param, **predict_params)[combination]
                    for combination, param, name in zip(feats_list, params_list, name_list)
                }

                recommendation = [result[name]['Position'].rename(name) for name in name_list]
                recommendation = pd.concat([result[name]['Position'].rename(name) for name in name_list], axis=1)
                # recommendation = recommendation.where(recommendation == 1, 'Short').where(recommendation == -1, 'Long')
                # recommendation = recommendation.iloc[-1].to_dict()
                # recommendation = (
                #     str(recommendation)
                #     .replace('{', '')
                #     .replace('}', '')
                #     .replace("'", "")
                #     .replace(', ', ' \n ')
                # )

                liquid_returns = [result[name]['Liquid_Return'].rename(name) for name in name_list]
                liquid_returns_concat = pd.concat(liquid_returns, axis=1)
                fig = GraphLayout(
                    liquid_returns_concat,
                    "BTC/USDT",
                    "1D",
                    "spot",
                ).grouped_lines()

            elif combination:
                param = get_model_feat_params(combination)
                result = (
                    model_predict
                    .calculate_model_returns(param, **predict_params)
                    [combination]
                )
                recommendation = result['Position'].to_frame()
                # recommendation = result['Position'].iloc[-1]
                # recommendation = "Short" if recommendation == -1 else "Long"
                graph_layout = GraphLayout(
                    result,
                    "BTC/USDT",
                    "1D",
                    "spot",
                )

                fig = graph_layout.plot_single_linechart("Liquid_Return")

            recommendation = recommendation.where(recommendation == 1, 'Short').where(recommendation == -1, 'Long')
            recommendation = recommendation.tail(5)
            recommendation = recommendation.reset_index()

            recommendation_table = dag.AgGrid(
                    rowData=recommendation.to_dict("records"),
                    getRowId="params.data.date",
                    columnDefs=[{"field": i} for i in recommendation.columns],
                    defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 115},
                    columnSize="sizeToFit",
                    dashGridOptions={"pagination": False},
                    className="ag-theme-alpine-dark",
                    style={"overflow": "hidden", "height": "40vh"},
                )

            return recommendation_table, fig

