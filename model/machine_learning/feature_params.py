from pydantic import BaseModel
import pandas as pd

class FeaturesParams(BaseModel):
    """
    A Pydantic BaseModel class for specifying parameters related to
    features in machine learning models.

    Attributes:
    -----------
    target_input : pd.Series
        The target input series for feature parameters.
    column : str
        The column name related to the feature.
    log_values : bool, optional
        Flag indicating whether to use log values
        (default: True).
    threshold : float, optional
        The threshold value for feature parameters
        (default: 0.5).
    higher_than_threshold : bool, optional
        Flag indicating whether values should be higher than the
        threshold
        (default: True).
    """
    target_input: pd.Series
    column: str
    log_values: bool = True
    threshold: float  = 0.5
    higher_than_threshold: bool = True

    class Config:
        arbitrary_types_allowed = True
