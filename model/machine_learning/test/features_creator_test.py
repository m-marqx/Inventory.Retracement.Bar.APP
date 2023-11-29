import unittest
import pandas as pd
import numpy as np

from model.machine_learning.utils import DataHandler
from model.machine_learning.features_creator import FeaturesCreator

class TestFeaturesCreator(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.read_parquet(
            'model/data/reference_dataset.parquet'
        )

        self.dataframe = DataHandler(self.dataframe).get_targets()
        self.return_series = self.dataframe['Return']
        self.source = self.dataframe['high'].pct_change() + 1
        self.validation_index = 70

        self.split_params = dict(
            target_input=self.dataframe['Target_1_bin'],
            column='temp_indicator',
            log_values=True,
        )

        self.split_paramsH = dict(
            target_input=self.dataframe['Target_1_bin'],
            column='temp_indicator',
            log_values=True,
            threshold=0.51,
        )
        self.split_paramsL = dict(
            target_input=self.dataframe['Target_1_bin'],
            column='temp_indicator',
            log_values=True,
            threshold=0.49,
            higher_than_threshold=False,
        )

        self.random_state = 42

        self.feature_creator = FeaturesCreator(
            self.dataframe,
            self.return_series,
            self.source,
            self.validation_index,
            self.split_params,
            self.split_paramsH,
            self.split_paramsL,
            self.random_state
        )
        self.model_params = {
            'objective' : "binary:logistic",
            'random_state' : self.random_state,
            'eval_metric' : 'auc'
        }

        self.dataframe = self.feature_creator.get_features(10)

        self.model_feature_creator = FeaturesCreator(
            self.dataframe,
            self.return_series,
            self.source,
            self.validation_index,
            self.split_params,
            self.split_paramsH,
            self.split_paramsL,
            self.random_state
        )

        features = [
            'temp_feature_RSI',
            'temp_feature_RSIH',
            'temp_feature_RSIL'
        ]

        target = 'Target_1_bin'

        self.model_results = self.model_feature_creator.train_model(
            features,
            target,
            self.model_params
        )

    def test_get_features(self) -> None:
        result = self.dataframe.iloc[-5:]

        referecence_values = {
        'open': [35741.65, 37408.35, 37294.27, 37713.57, 37780.67],
        'high': [37861.1, 37653.44, 38414.0, 37888.0, 37814.63],
        'low': [35632.01, 36870.0, 37251.51, 37591.1, 37150.0],
        'close': [37408.34, 37294.28, 37713.57, 37780.67, 37505.91],
        'Return': [
            1.0466315908750714,
            0.9969509473021257,
            1.0112427428549364,
            1.0017792004310384,
            0.9927274979506717
        ],
        'Target_1': [
            0.9969509473021257,
            1.0112427428549364,
            1.0017792004310384,
            0.9927274979506717,
            np.nan
        ],
        'Target_1_bin': [0.0, 1.0, 1.0, 0.0, np.nan],
        'temp_RSI': [
            50.488933517823305,
            47.80489985991004,
            54.07752991083207,
            45.976333452252014,
            48.925335617224185
        ],
        'temp_feature_RSI': [0, 0, 1, 0, 0],
        'temp_feature_RSIH': [0, 0, 1, 0, 0],
        'temp_feature_RSIL': [1, 0, 0, 1, 1],
        'date': [
            pd.Timestamp('2023-11-22 00:00:00'),
            pd.Timestamp('2023-11-23 00:00:00'),
            pd.Timestamp('2023-11-24 00:00:00'),
            pd.Timestamp('2023-11-25 00:00:00'),
            pd.Timestamp('2023-11-26 00:00:00')
        ]
        }

        temp_features = [
            'temp_feature_RSI',
            'temp_feature_RSIH',
            'temp_feature_RSIL'
        ]

        referecence_df = pd.DataFrame(referecence_values).set_index('date')

        referecence_df[temp_features] = (
            referecence_df[temp_features]
            .astype('int8')
        )

        pd.testing.assert_frame_equal(result, referecence_df)

    def test_features_generated(self):
        result_columns = list(self.dataframe.columns)
        reference_columns = [
            'open',
            'high',
            'low',
            'close',
            'Return',
            'Target_1',
            'Target_1_bin',
            'temp_RSI',
            'temp_feature_RSI',
            'temp_feature_RSIH',
            'temp_feature_RSIL'
        ]
        assert result_columns == reference_columns

    def test_model_returns(self) -> None:
        reference_keys = ['model', 'X_train', 'X_test', 'y_train', 'y_test']

        assert list(self.model_results.keys()) == reference_keys

    def test_X_Train(self) -> None:
        result = self.model_results['X_train'].iloc[-10:]

        train_indexes = [
            pd.Timestamp('2012-01-27 00:00:00'),
            pd.Timestamp('2012-01-28 00:00:00'),
            pd.Timestamp('2012-01-29 00:00:00'),
            pd.Timestamp('2012-01-30 00:00:00'),
            pd.Timestamp('2012-01-31 00:00:00'),
            pd.Timestamp('2012-02-01 00:00:00'),
            pd.Timestamp('2012-02-02 00:00:00'),
            pd.Timestamp('2012-02-03 00:00:00'),
            pd.Timestamp('2012-02-04 00:00:00'),
            pd.Timestamp('2012-02-05 00:00:00')
        ]

        reference_X_train = pd.DataFrame({
            'temp_feature_RSI': [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            'temp_feature_RSIH': [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            'temp_feature_RSIL': [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            'date': train_indexes
        }).set_index('date').astype('int8')

        pd.testing.assert_frame_equal(result, reference_X_train)

    def test_X_test(self) -> None:
        result = self.model_results['X_test'].iloc[-10:]

        test_indexes = [
            pd.Timestamp('2012-03-02 00:00:00'),
            pd.Timestamp('2012-03-03 00:00:00'),
            pd.Timestamp('2012-03-04 00:00:00'),
            pd.Timestamp('2012-03-05 00:00:00'),
            pd.Timestamp('2012-03-06 00:00:00'),
            pd.Timestamp('2012-03-07 00:00:00'),
            pd.Timestamp('2012-03-08 00:00:00'),
            pd.Timestamp('2012-03-09 00:00:00'),
            pd.Timestamp('2012-03-10 00:00:00'),
            pd.Timestamp('2012-03-11 00:00:00'),
        ]

        reference_X_test = pd.DataFrame({
            'temp_feature_RSI':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'temp_feature_RSIH': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'temp_feature_RSIL': [0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            'date': test_indexes
        }).set_index('date').astype('int8')

        pd.testing.assert_frame_equal(result, reference_X_test)

    def test_y_train(self) -> None:
        result = self.model_results['y_train'].iloc[-10:]

        train_indexes = [
            pd.Timestamp('2012-01-27 00:00:00'),
            pd.Timestamp('2012-01-28 00:00:00'),
            pd.Timestamp('2012-01-29 00:00:00'),
            pd.Timestamp('2012-01-30 00:00:00'),
            pd.Timestamp('2012-01-31 00:00:00'),
            pd.Timestamp('2012-02-01 00:00:00'),
            pd.Timestamp('2012-02-02 00:00:00'),
            pd.Timestamp('2012-02-03 00:00:00'),
            pd.Timestamp('2012-02-04 00:00:00'),
            pd.Timestamp('2012-02-05 00:00:00')
        ]

        reference_y_train = pd.Series(
            [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            index=train_indexes,
            name='Target_1_bin',
        ).rename_axis('date')

        pd.testing.assert_series_equal(result, reference_y_train)

    def test_y_test(self) -> None:
        result = self.model_results['y_test'].iloc[-10:]

        test_indexes = [
            pd.Timestamp('2012-03-02 00:00:00'),
            pd.Timestamp('2012-03-03 00:00:00'),
            pd.Timestamp('2012-03-04 00:00:00'),
            pd.Timestamp('2012-03-05 00:00:00'),
            pd.Timestamp('2012-03-06 00:00:00'),
            pd.Timestamp('2012-03-07 00:00:00'),
            pd.Timestamp('2012-03-08 00:00:00'),
            pd.Timestamp('2012-03-09 00:00:00'),
            pd.Timestamp('2012-03-10 00:00:00'),
            pd.Timestamp('2012-03-11 00:00:00'),
        ]

        reference_y_test = pd.Series(
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            index=test_indexes,
            name='Target_1_bin',
        ).rename_axis('date')

        pd.testing.assert_series_equal(result, reference_y_test)

    def test_get_results(self) -> None:
        features = [
            'temp_feature_RSI',
            'temp_feature_RSIH',
            'temp_feature_RSIL'
        ]

        result = (
            self.model_feature_creator
            .get_results(20, features, 'Target_1_bin').iloc[-5:]
        )

        reference_results_dict = {
        'y_pred_probs': [
            0.11044099926948547,
            0.5902235507965088,
            0.8577237725257874,
            0.11044099926948547,
            0.11044099926948547],

        'Period_Return': [
            0.046631590875071405,
            -0.003049052697874255,
            0.011242742854936427,
            0.001779200431038408,
            -0.007272502049328278
        ],
        'Predict': [-1, 1, 1, -1, -1],
        'Position': [1.0, -1.0, 1.0, 1.0, -1.0],
        'Result': [
            0.046631590875071405,
            0.003049052697874255,
            0.011242742854936427,
            0.001779200431038408,
            0.007272502049328278
        ],
        'Liquid_Result': [
            0.045631590875071404,
            0.002049052697874255,
            0.010242742854936426,
            0.000779200431038408,
            0.006272502049328278
        ],
        'Period_Return_cum': [
            12.761216884679476,
            12.758167831981602,
            12.769410574836538,
            12.771189775267576,
            12.763917273218247
        ],
        'Total_Return': [
            7.304220949472645,
            7.30727000217052,
            7.318512745025457,
            7.3202919454564945,
            7.327564447505823
        ],
        'Liquid_Return': [
            2.9972209494726862,
            2.9992700021705607,
            3.0095127450254973,
            3.010291945456536,
            3.016564447505864
        ],
        'max_Liquid_Return': [
            4.8310938840163065,
            4.8310938840163065,
            4.8310938840163065,
            4.8310938840163065,
            4.8310938840163065
        ],
        'drawdown': [
            0.37959786718511024,
            0.379173728729707,
            0.37705355820500974,
            0.3768922695921727,
            0.3755939089724214
        ],
        'drawdown_duration': [1042, 1043, 1044, 1045, 1046],
        'validation_date': [
            '2012-03-12 00:00:00',
            '2012-03-12 00:00:00',
            '2012-03-12 00:00:00',
            '2012-03-12 00:00:00',
            '2012-03-12 00:00:00'
        ],
        'date': [
            pd.Timestamp('2023-11-22 00:00:00'),
            pd.Timestamp('2023-11-23 00:00:00'),
            pd.Timestamp('2023-11-24 00:00:00'),
            pd.Timestamp('2023-11-25 00:00:00'),
            pd.Timestamp('2023-11-26 00:00:00')
        ]
        }

        reference_result = (
            pd.DataFrame(reference_results_dict)
            .set_index('date')
        )

        reference_result['y_pred_probs'] = (
        reference_result['y_pred_probs'].astype('float32')
        )
        reference_result['Predict'] = (
            reference_result['Predict']
            .astype('int32')
        )
        pd.testing.assert_frame_equal(reference_result,result)
