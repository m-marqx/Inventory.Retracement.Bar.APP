import unittest
import pandas as pd
import numpy as np
from model.machine_learning.utils import DataHandler

class TestDatasetManipulation(unittest.TestCase):
    def setUp(self):
        self.expected_dataset = (
            pd.read_parquet('model/data/reference_dataset.parquet').iloc[:-1]
        )

        self.updated_dataset = (
            pd.read_parquet('model/data/dataset_updated.parquet').iloc[:-1]
        )

    def test_dataset(self):
        updated_dataset = self.updated_dataset.reindex(
            self.expected_dataset.index
        ).copy()

        pd.testing.assert_frame_equal(self.expected_dataset, updated_dataset)

    def test_if_dataset_is_updated(self):
        assert self.updated_dataset.shape[0] > self.expected_dataset.shape[0]

    def test_get_target_method(self):
        dates = pd.DatetimeIndex([
            '2017-08-17',
            '2017-08-18',
            '2017-08-19',
            '2017-08-20',
            '2017-08-21'
        ])

        df = pd.DataFrame(
            {'close': [4285.08, 4108.37, 4139.98, 4086.29, 4016.00]},
            index=dates
        )

        result = DataHandler(df).get_targets()

        returns = [np.nan,
            0.9587615633780466,
            1.0076940489780617,
            0.9870313383156442,
            0.9827985776829349
        ]

        target_1 = [
            0.9587615633780466,
            1.0076940489780617,
            0.9870313383156442,
            0.9827985776829349,
            np.nan
        ]

        expected_result = pd.DataFrame({
            'close': [4285.08, 4108.37, 4139.98, 4086.29, 4016.00],
            'Return': returns,
            'Target_1': target_1,
            'Target_1_bin': [0.0, 1.0, 0.0, 0.0, np.nan]},
            index=dates)

        pd.testing.assert_frame_equal(result, expected_result)
