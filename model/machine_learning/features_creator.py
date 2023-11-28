import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import tradingview_indicators as ta

from model.machine_learning.utils import (
    DataHandler,
    ModelHandler
)

class FeaturesCreator:
    """
    A class for creating features, training XGBoost models, and
    obtaining results.

    This class handles the creation of features, training of an XGBoost
    model, and obtaining
    results based on the provided data.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data.
    return_series : pd.Series
        The return series.
    source : pd.Series
        The source series.
    validation_index : int
        The index for splitting the DataFrame into development and
        validation sets.
    split_params : dict
        Splitting parameters for the main feature.
    split_paramsH : dict
        Splitting parameters for the high feature.
    split_paramsL : dict
        Splitting parameters for the low feature.
    random_state : int, optional
        Random state for reproducibility (default: 42).

    Methods:
    --------
    get_features(value: int) -> pd.DataFrame:
        Get features for the Features_Creator.

    train_model(
    features: str | list, \
    target: str, \
    model_params: dict | None = None \
    ) -> xgb.XGBClassifier:
        Train an XGBoost model.

    get_results(value: int, \
    features: list, target: str, \
    result_column: str | None = None, \
    model_params: dict | None = None \
    ) -> pd.DataFrame:
        Get results from the trained XGBoost model.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        return_series: pd.Series,
        source: pd.Series,
        validation_index: int,
        split_params: dict,
        split_paramsH: dict,
        split_paramsL: dict,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the Features_Creator class.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input DataFrame containing the data.
        return_series : pd.Series
            The return series.
        source : pd.Series
            The source series.
        validation_index : int
            The index for splitting the DataFrame into development and
            validation sets.
        split_params : dict
            Splitting parameters for the main feature.
        split_paramsH : dict
            Splitting parameters for the high feature.
        split_paramsL : dict
            Splitting parameters for the low feature.
        random_state : int, optional
            Random state for reproducibility (default: 42).

        """
        self.data_frame = DataHandler(dataframe).get_targets()
        self.return_series = return_series
        self.source = source
        self.random_state = random_state
        self.test_size = 0.5

        validation_index = validation_index or int(dataframe.shape[0] * 0.7)

        self.development = (
                self.data_frame.iloc[:validation_index].copy()
                if isinstance(validation_index, int)
                else self.data_frame.loc[:validation_index].copy()
            )

        self.validation = (
            self.data_frame.iloc[validation_index:].copy()
            if isinstance(validation_index, int)
            else self.data_frame.loc[validation_index:].copy()
        )

        self.split_params = split_params
        self.split_paramsH = split_paramsH
        self.split_paramsL = split_paramsL

    def get_features(
        self,
        value: int,
    ) -> pd.DataFrame:
        """
        Get features for the Features_Creator.

        Parameters:
        -----------
        value : int
            The value for RSI calculation.

        Returns:
        --------
        pd.DataFrame
            DataFrame with added features.

        """
        train_development = int(self.development.shape[0] * self.test_size)

        self.data_frame['temp_RSI'] = ta.RSI(self.source, value)
        temp_RSI = self.data_frame['temp_RSI'].iloc[:train_development]
        data_handler = DataHandler(temp_RSI)

        intervals = (
            data_handler
            .get_split_variable_intervals(**self.split_params)
        )

        intervalsH = (
            data_handler
            .get_split_variable_intervals(**self.split_paramsH)
        )

        intervalsL = (
            data_handler
            .get_split_variable_intervals(**self.split_paramsL)
        )

        self.data_frame['temp_feature_RSI'] = (
            DataHandler(self.data_frame)
            .get_intervals_variables('temp_RSI', intervals)
        ).astype('int8')

        self.data_frame['temp_feature_RSIH'] = (
            DataHandler(self.data_frame)
            .get_intervals_variables('temp_RSI', intervalsH)
        ).astype('int8')

        self.data_frame['temp_feature_RSIL'] = (
            DataHandler(self.data_frame)
            .get_intervals_variables('temp_RSI', intervalsL)
        ).astype('int8')

        return self.data_frame

    def train_model(
        self,
        features: str | list,
        target: str,
        model_params: dict | None = None,
    ) -> xgb.XGBClassifier:
        """
        Train an XGBoost model.

        Parameters:
        -----------
        features : str | list
            Features used for training.
        target : str
            Target variable.
        model_params : dict, optional
            Parameters for the XGBoost model
            (default: None).

        Returns:
        --------
        xgb.XGBClassifier
            Trained XGBoost model.

        """

        if isinstance(target, list):
            raise ValueError('Target must be a string')

        if not model_params:
            model_params = {
                'objective' : "binary:logistic",
                'random_state' : self.random_state,
                'eval_metric' : 'auc'
            }

        X_train, X_test, y_train, y_test = train_test_split(
            self.development[features],
            self.development[target],
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=False
        )

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        model_dict = {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        return model_dict

