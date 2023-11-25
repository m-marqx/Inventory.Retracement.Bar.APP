import pandas as pd
from model.utils.enums import ReturnType
from model.utils.exceptions import InvalidArgumentError


class MathFeature:
    """
    A class for calculating mathematical features based on the input
    data.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing the data.
    source_column : str
        The name of the column representing the price values.
    feat_last_column : str, optional
        The name of the column representing the last feature
        (default: None).
    return_type : Literal["short", "full"], optional
        The return type of methods ('short' returns only calculated
        values, 'full' returns the modified DataFrame with added
        columns).
        (default: "short")

    Attributes
    ----------
    dataframe : pandas.DataFrame
        The copy of the input dataframe.
    source_column : str
        The name of the column representing the price values.
    feat_last_column : str
        The name of the column representing the last feature.
    return_type : Literal["short", "full"]
        The return type of methods.

    Methods
    -------
    rolling_ratio(fast_length, slow_length, method)
        Calculate a rolling ratio of two rolling averages.

    ratio(length, method)
        Compute ratio-based features.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        source_column: str,
        feat_last_column: str = None,
        return_type: ReturnType = ReturnType.SHORT,
    ) -> None:
        """
        Initialize the MathVariables class.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input dataframe containing the data.
        source_column : str
            The name of the column representing the price values.
        feat_last_column : str, optional
            The name of the column representing the last feature
            (default: None).
        return_type : Literal["short", "full"], optional
            The return type of methods ('short' returns only calculated
            values, 'full' returns the modified DataFrame with added
            columns).
            (default: "short")
        """
        self.dataframe = dataframe.copy()
        self.source_column = source_column
        self.feat_last_column = feat_last_column
        self.return_type = return_type
        return_types = set(operation for operation in ReturnType)

        if return_type not in return_types:
            raise InvalidArgumentError(f"{return_type} not found")
