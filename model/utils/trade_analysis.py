from typing import Literal
import logging
import time

import pandas as pd
import numpy as np
import ccxt


class TradeAnalysis:
    """
    A class for analyzing trade data from cryptocurrency exchanges.

    Attributes:
    -----------
    exchange : ccxt.Exchange
        The cryptocurrency exchange object.
    symbol : str
        The trading symbol (e.g., 'BTC/USDT').
    market_type : str
        The market type ('spot', 'margin', etc.).
    main_currency : str
        The main currency used for fee calculations.
    start_time : int
        The start time in milliseconds.
    end_time : int, optional
        The end time in milliseconds. If not provided, the current time
        is used.
    verbose : bool, optional
        If True, log messages will be displayed. Default is False.
    **kwargs : dict
        Additional parameters for fetching trades.

    Methods:
    --------
    __get_trades_from_futures() -> list:
        Fetch trades for futures market type.

    get_trades(colored: bool = False) -> pd.DataFrame:
        Fetch and process trades data.

    calculate_trade_results(dataframe: pd.DataFrame, \
    first_side: Literal['buy', 'sell'] = 'buy') -> pd.DataFrame:
        Calculate trade results.

    get_and_caculate_trades() -> pd.DataFrame:
        Fetch and calculate trade results.

    caculate_trade_daily_results() -> pd.DataFrame:
        Calculate daily trade results.
    """

    def __init__(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        market_type: str,
        main_currency: str,
        start_time: int,
        end_time: int = None,
        verbose: bool = False,
        **kwargs
    ) -> None:
        """
        A class for analyzing trade data from cryptocurrency
        exchanges.

        Parameters:
        -----------
        exchange : ccxt.Exchange
            The cryptocurrency exchange object.
        symbol : str
            The trading symbol (e.g., 'BTC/USDT').
        market_type : str
            The market type ('spot', 'margin', etc.).
        main_currency : str
            The main currency used for fee calculations.
        start_time : int
            The start time in milliseconds.
        end_time : int, optional
            The end time in milliseconds. If not provided, the current
            time is used.
        verbose : bool, optional
            If True, log messages will be displayed. Default is False.
        **kwargs : dict
            Additional parameters for fetching trades.

        """
        self.exchange = exchange
        self.symbol = symbol
        self.start_time = start_time
        self.market_type = market_type
        self.kwargs = kwargs
        self.main_currency = main_currency

        if verbose:
            logging.basicConfig(
                format='%(levelname)s %(asctime)s: %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO,
                force=True,
            )
        else:
            logging.basicConfig(
                format='%(levelname)s %(asctime)s: %(message)s',
                datefmt='%H:%M:%S',
                level=logging.CRITICAL,
                force=True,
            )


        if not end_time:
            self.end_time = time.time() * 1000
        else:
            self.end_time = end_time

    def __get_trades_from_futures(self) -> list:
        """
        Fetch trades for futures market type.

        Returns:
        --------
        list
            List of trades.
        """
        day = 24 * 60 * 60 * 1000
        total_days = (self.end_time - self.start_time) / day

        all_trades = []
        range_time = range(self.start_time, self.end_time, day)

        for index, start_time in enumerate(range_time):
            end_time = start_time + day
            load_percentage = (index / total_days) * 100

            trades = self.exchange.fetch_my_trades(
                self.symbol,
                start_time,
                None,
                {'endTime': end_time, **self.kwargs}
            )

            if len(trades):
                last_trade = trades[-1]
                self.start_time = last_trade['timestamp'] + 1
                all_trades += trades
            else:
                self.start_time = end_time

            logging.info("Processing trades [%.2f%%]", load_percentage)

            if len(all_trades) > 0:
                logging.info("Trades processed: %s\n", len(all_trades))

        return all_trades

    def get_trades(self, colored: bool = False) -> pd.DataFrame:
        """
        Fetch and process trades data.

        Parameters:
        -----------
        colored : bool, optional
            If True, the DataFrame is styled with colors based on the
            'side'.
            (default: False).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing trades data.
        """
        trade_params = {
            'type': self.market_type,
            **self.kwargs
        }
        if not trade_params.get('endTime', None) and self.end_time:
            trade_params['endTime'] = self.end_time

        if self.market_type in ['spot', 'margin']:
            trades = self.exchange.fetch_my_trades(
                symbol=self.symbol,
                since=self.start_time,
                params=trade_params
            )
        else:
            trades = self.__get_trades_from_futures()

        logging.info("Processing trades [100%]")
        logging.info("Trades processed: %s\n", len(trades))

        if not trades:
            raise ValueError('No trades found')

        trades_df = (
            pd.DataFrame(trades)
            .drop(columns=['info', 'id', 'datetime', 'order', 'type', 'fees'])
            .set_index('timestamp')
        )
        trades_df.index = pd.to_datetime(trades_df.index, unit='ms')
        trades_df = trades_df.rename_axis('date')
        trades_df = trades_df.rename(columns={'cost': 'amount_quote'})

        fee = pd.DataFrame(list(trades_df['fee'])).set_index(trades_df.index)
        fee.columns = ['fee_cost', 'fee_currency']
        trades_df = pd.concat([trades_df, fee], axis=1).drop(columns='fee')
        trades_df['fee_cost_USD'] = np.where(
            trades_df['fee_currency'] == self.main_currency,
            trades_df['price'] * trades_df['fee_cost'], trades_df['fee_cost']
        )

        if colored:
            trades_df = trades_df.reset_index().style.apply(
                lambda row: [
                    "color: #FF3344" if row['side'] == 'sell'
                    else "color: #00e676" for _ in row
                ], axis=1
            )

        return trades_df

    def calculate_trade_results(
        self,
        dataframe: pd.DataFrame,
        first_side: Literal['buy', 'sell'] = 'buy'
    ) -> pd.DataFrame:
        """
        Calculate trade results.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            DataFrame containing trades data.
        first_side : Literal['buy', 'sell'], optional
            The first side of the trade.
            (default: 'buy')

        Returns:
        --------
        pd.DataFrame
            DataFrame containing calculated trade results.
        """
        second_side = 'sell' if first_side == 'buy' else 'buy'

        analysis_df = pd.DataFrame(index=dataframe.index)
        analysis_df['open_price'] = np.where(
            dataframe['side'] == first_side,
            dataframe['price'], np.nan
        )

        analysis_df['close_price'] = np.where(
            dataframe['side'] == second_side,
            dataframe['price'], np.nan
        )
        analysis_df['close_price'] = analysis_df['close_price'].bfill()
        analysis_df['amount_quote'] = dataframe['amount_quote']
        analysis_df['fee_cost'] = dataframe['fee_cost_USD']

        analysis_df['result'] = -(
            (
                (analysis_df['close_price'] - analysis_df['open_price'])
                / analysis_df['open_price']
            ) * 100
        )

        analysis_df['total_fee'] = np.where(
            analysis_df['result'].isna(),
            np.nan, analysis_df['fee_cost'] + analysis_df['fee_cost'].shift(-1)
        )
        analysis_df['diff_quote'] = np.where(
            analysis_df['result'].isna(),
            np.nan, -(analysis_df['amount_quote'].diff(-1))
        )

        analysis_df.fillna('')
        return analysis_df

    def calculate_trade_daily_results(self) -> pd.DataFrame:
        """
        Calculate daily trade results.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing calculated daily trade results.
        """
        sum_columns = ['amount', 'amount_quote', 'fee_cost', 'fee_cost_USD']
        drop_columns = ['takerOrMaker', 'symbol', 'fee_currency']

        trades_df = self.get_trades()
        trades_df['amount'] = np.where(
            trades_df['side'] == 'buy',
            trades_df['amount'],
            -abs(trades_df['amount'])
        )
        trades_df = trades_df.drop(columns=drop_columns)

        price_serie = (
            trades_df
            .groupby([trades_df.index.date, 'side'])['price']
            .mean()
        )

        sum_df = (
            trades_df
            .groupby([trades_df.index.date, 'side'])[sum_columns]
            .sum()
        )

        result_df = (
            pd.concat([price_serie, sum_df], axis=1)
            .reset_index()
            .set_index('level_0')
            .rename_axis('date')
        )

        first_side = result_df['side'][0]

        analysis_df = self.calculate_trade_results(result_df, first_side)
        return analysis_df
