import unittest
import pandas as pd
from model.machine_learning.utils import DataHandler
from model.machine_learning.features_creator import FeaturesCreator
from model.machine_learning.feature_params import (
    FeaturesParams,
    FeaturesParamsComplete,
)

class TestFeaturesCreator(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.read_parquet('model/data/reference_dataset.parquet')
        self.dataframe = DataHandler(self.dataframe).calculate_targets()
        split_params = FeaturesParams(
            target_input=self.dataframe['Target_1_bin'],
            column='temp_indicator',
            log_values=True,
        )

        split_paramsH = FeaturesParams(
            target_input=self.dataframe['Target_1_bin'],
            column='temp_indicator',
            log_values=True,
            threshold=0.52,
        )

        split_paramsL = FeaturesParams(
            target_input=self.dataframe['Target_1_bin'],
            column='temp_indicator',
            log_values=True,
            threshold=0.48,
            higher_than_threshold=False,
        )

        feature_params = FeaturesParamsComplete(
            split_features = split_params,
            high_features = split_paramsH,
            low_features = split_paramsL,
        )

        self.return_series = self.dataframe['close'].pct_change()
        self.source = self.dataframe['close'].pct_change().dropna()
        self.validation_index = '2020-01-01 00:00:00'
        self.features_creator = FeaturesCreator(
            dataframe=self.dataframe[['open', 'high', 'low', 'close']],
            return_series=self.return_series,
            source=self.source,
            feature_params=feature_params,
            validation_index=self.validation_index
        )

    def test_rsi_result_creation(self) -> None:
        rsi_results = self.features_creator.calculate_model_returns(
            14, 'RSI'
        )
        rsi_results = pd.concat(rsi_results, axis=1)

        rsi_reference_results = pd.read_parquet(
            'model/machine_learning/test/rsi_reference_results_14.parquet'
        )

        pd.testing.assert_frame_equal(rsi_results, rsi_reference_results)

    def test_rolling_ratio_result_creation(self):
        ratio_results = self.features_creator.calculate_model_returns(
            [14, 28, 'std'], 'rolling_ratio'
        )

        ratio_results = pd.concat(ratio_results, axis=1)

        ratio_reference_results = pd.read_parquet(
            'model/machine_learning/test/rolling_rsi_ratio_reference_results_1428std.parquet'
        )

        pd.testing.assert_frame_equal(ratio_results, ratio_reference_results)

    def test_rsi_indicator(self):
        rsi_indicator = self.features_creator.temp_indicator(
            14, 'RSI', self.source,
        )
        rsi_indicator = rsi_indicator.to_frame()

        rsi_reference_indicator = pd.read_parquet(
            'model/machine_learning/test/rsi_reference_indicator_14.parquet'
        )

        pd.testing.assert_frame_equal(rsi_indicator, rsi_reference_indicator)

    def test_rolling_ratio_indicator(self):
        rolling_rsi_ratio_indicator = self.features_creator.temp_indicator(
            [14, 28, 'std'], 'rolling_ratio'
        )

        rolling_rsi_ratio_indicator = rolling_rsi_ratio_indicator.to_frame()

        ratio_reference_indicator = pd.read_parquet(
            'model/machine_learning/test/rolling_rsi_ratio_reference_indicator_1428std.parquet'
        )

        pd.testing.assert_frame_equal(rolling_rsi_ratio_indicator, ratio_reference_indicator)

    def test_get_results(self):
        self.features_creator.data_frame['RSI'] = self.features_creator.temp_indicator(
            14, 'RSI', self.source,
        )

        self.features_creator.calculate_features("RSI", 1526)

        all_features_results = (
            self.features_creator
            .calculate_results(['RSI_split','RSI_high','RSI_low'])
        )

        all_features_reference_results = pd.read_parquet(
            'model/machine_learning/test/rsi_reference_get_results_14_all_features.parquet'
        )

        pd.testing.assert_frame_equal(all_features_results, all_features_reference_results)
