import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from model.utils.trade_analysis import TradeAnalysis

class TestTradeAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.exchange = MagicMock()
        self.symbol = 'BTC/USDT'
        self.market_type = 'spot'
        self.main_currency = 'BTC'
        self.start_time = 1630000000000
        self.end_time = 1631000000000
        self.verbose = False
        self.kwargs = {}
        self.trade_analysis = TradeAnalysis(
            self.exchange,
            self.symbol,
            self.market_type,
            self.main_currency,
            self.start_time,
            self.end_time,
            self.verbose
        )

        self.mock_trades = [
            {
                'info': {
                    'symbol': 'BTCUSD_PERP',
                    'id': '255339169',
                    'orderId': '24120305033',
                    'pair': 'BTCUSD',
                    'side': 'BUY',
                    'price': '41000',
                    'qty': '460',
                    'realizedPnl': '0',
                    'marginAsset': 'BTC',
                    'baseQty': '1.122',
                    'commission': '0.005615',
                    'commissionAsset': 'BTC',
                    'time': '1620698679123',
                    'positionSide': 'BOTH',
                    'buyer': True,
                    'maker': False
                    },
                'timestamp': 1620698679123,
                'datetime': '2021-05-10T23:04:39.123000Z',
                'symbol': 'BTC/USD:BTC',
                'id': '255339169',
                'order': '24120305033',
                'type': None,
                'side': 'buy',
                'takerOrMaker': 'taker',
                'price': 41000.0,
                'amount': 460.0,
                'cost': 1.122,
                'fee': {'cost': 0.005615, 'currency': 'BTC'},
                'fees': [{'cost': 0.005615, 'currency': 'BTC'}]
            },
            {
                'info': {
                    'symbol': 'BTCUSD_PERP',
                    'id': '149541422',
                    'orderId': '23151000201',
                    'pair': 'BTCUSD',
                    'side': 'SELL',
                    'price': '42000',
                    'qty': '460',
                    'realizedPnl': '0.0267',
                    'marginAsset': 'BTC',
                    'baseQty': '1.2467',
                    'commission': '0.0062335',
                    'commissionAsset': 'BTC',
                    'time': '1620882279123',
                    'positionSide': 'BOTH',
                    'buyer': False,
                    'maker': False
                },
                'timestamp': 1620882279123,
                'datetime': '2021-05-13T02:04:39.123000Z',
                'symbol': 'BTC/USD:BTC',
                'id': '149541422',
                'order': '23151000201',
                'type': None,
                'side': 'sell',
                'takerOrMaker': 'taker',
                'price': 42000,
                'amount': 460.0,
                'cost': 1.2467,
                'fee': {'cost': 0.0062335, 'currency': 'BTC'},
                'fees': [{'cost': 0.0062335, 'currency': 'BTC'}]
            }
        ]
        self.exchange.fetch_my_trades = MagicMock(
            return_value=self.mock_trades
        )

    def test_get_trades(self):
        drop_columns = ['info', 'id', 'datetime', 'order', 'type', 'fees']

        expected_trades_df = (
            pd.DataFrame.from_records(self.mock_trades)
            .set_index('timestamp')
        )

        expected_trades_df.index = pd.to_datetime(
            expected_trades_df.index, unit='ms'
        )

        expected_trades_df = expected_trades_df.rename(
            columns={'cost': 'amount_quote'}
        )

        expected_trades_df_fee = (
            pd.DataFrame(list(expected_trades_df['fee']))
            .set_index(expected_trades_df.index)
        )

        expected_trades_df_fee.columns = ['fee_cost', 'fee_currency']

        expected_trades_df = (
            pd.concat([expected_trades_df, expected_trades_df_fee], axis=1)
            .drop(columns='fee')
        )

        expected_trades_df['fee_cost_USD'] = [230.215, 261.807]
        expected_trades_df = expected_trades_df.drop(columns=drop_columns)
        expected_trades_df = expected_trades_df.rename_axis('date')

        test_df = self.trade_analysis.get_trades()

        pd.testing.assert_frame_equal(test_df, expected_trades_df)

    def test_calculate_trade_results(self):
        test_trades = self.trade_analysis.get_trades()

        test_df = (
            self.trade_analysis
            .calculate_trade_results(test_trades, 'buy')
        )

        expected_dict = {
            'date': {
                0: pd.Timestamp('2021-05-11 02:04:39.123000'),
                1: pd.Timestamp('2021-05-13 05:04:39.123000')
            },
            'open_price': {0: 41000.0, 1: np.nan},
            'close_price': {0: 42000.0, 1: 42000.0},
            'amount_quote': {0: 1.122, 1: 1.2467},
            'fee_cost': {0: 230.215, 1: 261.807},
            'result': {0: -2.4390243902439024, 1: np.nan},
            'total_fee': {0: 492.02200000000005, 1: np.nan},
            'diff_quote': {0: 0.12469999999999981, 1: np.nan}
        }

        expected_df = pd.DataFrame(expected_dict).set_index('date')
        pd.testing.assert_frame_equal(test_df, expected_df)

    def test_caculate_trade_daily_results(self):
        expected_df = pd.DataFrame({
            'date': [
                pd.Timestamp('2021-05-11 02:04:39.123000'),
                pd.Timestamp('2021-05-13 05:04:39.123000')
            ],
            'open_price': [41000.0, np.nan],
            'close_price': [42000.0, 42000.0],
            'amount_quote': [1.122, 1.2467],
            'fee_cost': [230.215, 261.807],
            'result': [-2.4390243902439024, np.nan],
            'total_fee': [492.02200000000005, np.nan],
            'diff_quote': [0.12469999999999981, np.nan]
        }).set_index('date')
        expected_df.index = expected_df.index.date
        expected_df = expected_df.rename_axis('date')

        test_df = self.trade_analysis.calculate_trade_daily_results()
        pd.testing.assert_frame_equal(test_df, expected_df)
