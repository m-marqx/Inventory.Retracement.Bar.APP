import unittest
import pandas as pd
from model.machine_learning.utils import DataHandler
from model.machine_learning.feature_params import (
    FeaturesParams,
    FeaturesParamsComplete
)

class TestFeaturesParams(unittest.TestCase):
    def setUp(self):
        self.data_frame = pd.read_parquet(
            "model/data/reference_dataset.parquet"
        )

        self.data_frame = DataHandler(self.data_frame).get_targets()

    def test_split_params(self):
        split_params = dict(
            target_input=self.data_frame["Target_1_bin"],
            column="temp_indicator",
            log_values = True,
            threshold = 0.5,
            higher_than_threshold = True
        )

        split_paramsH = dict(
            target_input=self.data_frame["Target_1_bin"],
            column="temp_indicator",
            log_values = True,
            threshold = 0.51,
            higher_than_threshold = True
        )

        split_paramsL = dict(
            target_input=self.data_frame["Target_1_bin"],
            column="temp_indicator",
            log_values=True,
            threshold=0.49,
            higher_than_threshold=False,
        )

        reference_split_params = FeaturesParams(
            target_input=self.data_frame["Target_1_bin"],
            column="temp_indicator",
            log_values=True,
        )

        reference_split_paramsH = FeaturesParams(
            target_input=self.data_frame["Target_1_bin"],
            column="temp_indicator",
            log_values=True,
            threshold=0.51,
        )

        reference_split_paramsL = FeaturesParams(
            target_input=self.data_frame["Target_1_bin"],
            column="temp_indicator",
            log_values=True,
            threshold=0.49,
            higher_than_threshold=False,
        )

        complete_params = FeaturesParamsComplete(
            split_features=split_params,
            high_features=split_paramsH,
            low_features=split_paramsL,
        )

        self.assertDictEqual(split_params, reference_split_params.dict())
        self.assertDictEqual(split_paramsH, reference_split_paramsH.dict())
        self.assertDictEqual(split_paramsL, reference_split_paramsL.dict())
        self.assertDictEqual(complete_params.split_features.dict(), reference_split_params.dict())
        self.assertDictEqual(complete_params.high_features.dict(), reference_split_paramsH.dict())
        self.assertDictEqual(complete_params.low_features.dict(), reference_split_paramsL.dict())
