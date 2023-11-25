from enum import Enum


class ReturnType(Enum):
    """
    Enum class defining return types for methods.

    Attributes
    ----------
    SHORT : str
        Represents the 'short' return type, which returns only
        calculated values.
    FULL : str
        Represents the 'full' return type, which returns the modified
        DataFrame with added columns.
    """
    SHORT = 'short'
    FULL = 'full'
