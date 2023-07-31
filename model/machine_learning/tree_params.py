from pydantic import BaseModel
import pandas as pd
import numpy as np

class TreeParams(BaseModel):
    n_estimators: list = [1, 10, 50]
    max_depths: np.ndarray = np.linspace(1, 10, 10, endpoint=True).astype(int)
    min_samples_splits: np.ndarray = np.linspace(0.1, 1, 25, endpoint=True)
    min_samples_leafs: np.ndarray = np.linspace(0.1, 0.5, 25, endpoint=True)
    criterion: str = "gini"
    max_features: str = "sqrt"
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0
    bootstrap: bool = True
    oob_score: bool = False
    random_state: int = 42
    verbose: int = 0
    class Config:
        arbitrary_types_allowed = True


class TrainTestSplits(BaseModel):
    x_train: np.ndarray | pd.Series | pd.DataFrame | None = None
    x_test: np.ndarray | pd.Series | pd.DataFrame | None = None
    y_train: np.ndarray | pd.Series | pd.DataFrame | None = None
    y_test: np.ndarray | pd.Series | pd.DataFrame | None = None
    class Config:
        arbitrary_types_allowed = True
