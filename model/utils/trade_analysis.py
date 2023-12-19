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
