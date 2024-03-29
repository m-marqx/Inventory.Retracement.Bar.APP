import pandas as pd

def reorder_columns(
    self,
    reference_column: str,
    column_to_move: str | list[str] | pd.Index,
) -> pd.DataFrame:
    """Reorder columns in a DataFrame.

    Moves the specified column(s) to a new position in the DataFrame, just
    before the specified reference column.

    Parameters
    ----------
    self : pandas.DataFrame
        The input DataFrame.
    reference_column : str
        The name of the reference column.
    column_to_move : str, list of str, or pandas.Index
        The name(s) or index of the column(s) to be moved.

    Returns
    -------
    pandas.DataFrame
        The reordered DataFrame.

    Raises
    ------
    ValueError
        If an invalid column name or index is provided.

    """
    insert_position = self.columns.get_loc(reference_column)

    if isinstance(column_to_move, str):
        column_to_move = [column_to_move]

    if set(column_to_move).issubset(self.columns):
        remaining_cols = list(
            self.drop(column_to_move, axis=1)
            .columns
        )
    else:
        raise ValueError("Invalid column name or index.")

    columns_adjusted = (
        remaining_cols[:insert_position]
        + list(column_to_move)
        + remaining_cols[insert_position:]
    )

    return self[columns_adjusted]

pd.DataFrame.reorder_columns = reorder_columns

class ResampleOHLC(pd.DataFrame):
    """
    A subclass of pd.DataFrame for resampling OHLC
    (open, high, low, close) data.

    Parameters
    ----------
    period : str or DateOffset
        The resampling period for the OHLC data.
    *args
        Positional arguments to be passed to the pd.DataFrame
        constructor.
    **kwargs
        Keyword arguments to be passed to the pd.DataFrame
        constructor.

    Attributes
    ----------
    period : str or DateOffset
        The resampling period for the OHLC data.

    Methods
    -------
    ohlc()
        Resamples the data and returns the OHLC values.
    OHLC()
        Resamples the data and returns the OHLC values
        (alternative column names).

    Notes
    -----
    The Klines values can be different from Binance when compared,
    especially when the Klines timeframe is too short, for example,
    `1 minute x 2 hours`. This is because Binance only considers
    the opening opening price when the candle has at least one trade.
    If a candle doesn't have at least one trade, that price doesn't
    exist for it.

    """

    def __init__(self, period, *args, **kwargs):
        """
        Initialize the ResampleOHLC class.

        Parameters
        ----------
        period : str or DateOffset
            The resampling period for the OHLC data.
        *args
            Positional arguments to be passed to the pd.DataFrame
            constructor.
        **kwargs
            Keyword arguments to be passed to the pd.DataFrame
            constructor.

        """
        super().__init__(*args, **kwargs)
        self.period = period

    def ohlc(self, drop_na=True):
        """
        Resamples the data and returns the OHLC values.

        Returns
        -------
        pd.DataFrame
            The resampled data with OHLC values.

        Raises
        ------
        ValueError
            If no OHLC columns are found in the data.

        """
        ohlc_columns = ["open", "high", "low", "close"]

        ohlc = all(
            column in self.columns
            for column in ohlc_columns
        )

        if ohlc:
            self[ohlc_columns] = self.resample(
                self.period, origin="start").agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                })
            if drop_na:
                return self.dropna()
            return self
        raise ValueError("No OHLC columns found")

    def OHLC(self, drop_na=True):
        """
        Resamples the data and returns the OHLC values
        (alternative column names).

        Returns
        -------
        pd.DataFrame
            The resampled data with OHLC values.

        Raises
        ------
        ValueError
            If no OHLC columns are found in the data.

        """
        OHLC_columns = ["Open", "High", "Low", "Close"]

        OHLC = all(
            column in self.columns
            for column in OHLC_columns
        )

        if OHLC:
            self[OHLC_columns] = self.resample(
                self.period, origin="start").agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                })
            if drop_na:
                return self.dropna()
            return self
        raise ValueError("No OHLC columns found")

def resample_ohlc_wrapper(self, period):
    """
    Wrapper function for creating a ResampleOHLC object.

    Parameters
    ----------
    period : str or DateOffset
        The resampling period for the OHLC data.

    Returns
    -------
    ResampleOHLC
        The ResampleOHLC object.

    """
    return ResampleOHLC(period, data=self.copy())

pd.DataFrame.resample_ohlc = resample_ohlc_wrapper
