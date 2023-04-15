import numpy as np
from pydantic import BaseModel
from typing import List


class EmaParamsBacktest(BaseModel):
    length: List[int] = np.arange(1, 100 + 1, 1)
    source_column: List[str] = ["open", "high", "low", "close"]


