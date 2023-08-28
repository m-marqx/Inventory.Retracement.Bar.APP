import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import graphviz

from tree_params import TreeParams, TrainTestSplits

class DataPreprocessor:
    """
    A class for preprocessing and splitting data for binary
    classification tasks.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data.
    features : str or list of str, optional
        The list of feature column names to use in the DataFrame.
        If None, defaults to ["IRB_Condition", "Signal", "uptrend"].

    Attributes:
    -----------
    dataframe : pd.DataFrame
        The DataFrame containing the data.
    features : list of str
        The list of feature column names used in the DataFrame.

    Methods:
    --------
    get_target_bin(periods=12, column="close", cutoff=None) ->
    pd.DataFrame
        Generate a target binary DataFrame based on a specified column.

    get_splits(target) ->
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Split the DataFrame into training and testing sets.

    tree_view(target, fitted_tree) -> None
        Visualize the decision tree using Graphviz.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: str | list[str] = None,
    ) -> None:
        """
        Initialize the DataPreprocessor object.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input DataFrame containing the data.
        features : str or list of str, optional
            The list of feature column names to use in the DataFrame.
            If None, defaults to ["IRB_Condition", "Signal", "uptrend"].
        """
        self.dataframe = dataframe.copy()

        if features is None:
            features = ["IRB_Condition", "Signal", "uptrend"]

        self.features = list(features)

    def get_target_bin(
        self,
        periods: int = 12,
        column: str = "close",
        cutoff: float | list[float] = None,
    ) -> pd.DataFrame:
        """
        Generate a target binary DataFrame based on a specified column.

        Parameters:
        -----------
        periods : int, optional
            The number of periods used to calculate the percentage
            change.
            (default: 12)
        column : str, optional
            The column name from which the target binary is generated.
            (default: "close")
        cutoff : float or list of float, optional
            The cutoff value(s) to determine the target binary values.
            If a list of float values is provided, multiple target bins
            will be generated.
            If None, defaults to [0.25, 0.30, 0.35]

        Returns:
        --------
        pd.DataFrame
            The DataFrame with binary targets.
        """
        if cutoff is None:
            cutoff = [0.25, 0.30, 0.35]

        features = [column] + list(self.features)
        data_frame = self.dataframe[features].astype("float64").copy()

        data_frame["return"] = data_frame[column].pct_change(periods)
        data_frame["target"] = data_frame["return"].shift(-periods)
        data_frame["target_bin"] = np.where(data_frame["target"] > 0, 1, -1)

        for value in enumerate(cutoff):
            cutoff_value = data_frame["target"].std() * value[1]
            cutoff_range = np.logical_or(
                data_frame["target"] > cutoff_value,
                data_frame["target"] < -cutoff_value,
            )

            data_frame[f"Target_Bin{value[0]}"] = data_frame["target_bin"]

            data_frame[f"Target_Bin{value[0]}"] = np.where(
                cutoff_range,
                data_frame[f"Target_Bin{value[0]}"],
                0,
            )

        return data_frame

    def tree_view(
        self,
        target: pd.Series,
        fitted_tree: RandomForestClassifier,
    ) -> None:
        """
        Visualize the decision tree using Graphviz.

        Parameters:
        -----------
        target : pd.Series
            The target column to be used in the decision tree
            visualization.
        fitted_tree : RandomForestClassifier
            The fitted RandomForestClassifier model containing
            the decision tree.

        Returns
        -------
        None
            The tree visualization is displayed and saved as a PNG
            image.
        """
        tree = fitted_tree.estimators_[0]
        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=self.features,
            class_names=target.astype("str"),
            filled=True,
            rounded=True,
            special_characters=True,
        )

        graphviz.Source(dot_data, format="png").view()

class RandomForestSearcher:
    """
    Class for performing grid search using RandomForestClassifier.

    Parameters:
    -----------
    tree_params : TreeParams
        An object containing tree-related hyperparameters
        for RandomForestClassifier.
    splits : TrainTestSplits
        An object containing training and testing data splits.

    Attributes:
    -----------
    tree_params : dict
        Dictionary containing tree-related hyperparameters.
    splits : dict
        Dictionary containing training and testing data splits.

    Methods:
    --------
    analyze_result(params: dict | TrainTestSplits) -> list
        Fit and evaluate RandomForestClassifier with specified
        hyperparameters.

    run_grid_search(n_jobs=-1, verbose=1) -> pd.DataFrame
        Perform grid search and return results in a DataFrame.
    """

    def __init__(
        self,
        tree_params: TreeParams,
        splits: TrainTestSplits,
    ) -> None:
        """
        Initialize the RandomForestSearcher object.

        Parameters:
        -----------
        tree_params : TreeParams
            An object containing tree-related hyperparameters for
            RandomForestClassifier.
        splits : TrainTestSplits
            An object containing training and testing data splits.

        Returns:
        --------
        None
        """
        self.tree_params = dict(tree_params)
        self.splits = dict(splits)

    def analyze_result(
        self,
        params: dict | TrainTestSplits,
    ) -> list:
        """
        Fit and evaluate RandomForestClassifier with specified
        hyperparameters.

        Parameters:
        -----------
        params : dict or TrainTestSplits
            Dictionary containing hyperparameters for
            RandomForestClassifier.

        Returns:
        --------
        list
            A list of evaluation results for each target column.
        """
        params = dict(params)

        random_forest = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leafs"],
            min_samples_split=params["min_samples_splits"],
            criterion=self.tree_params["criterion"],
            max_features=self.tree_params["max_features"],
            n_jobs=1,
            bootstrap=self.tree_params["bootstrap"],
            oob_score=self.tree_params["oob_score"],
            random_state=self.tree_params["random_state"],
        )

        target_columns = self.splits["y_train"].columns
        results_dict = {column: [] for column in target_columns}

        for target_parallel in target_columns:
            random_forest.fit(
                self.splits["x_train"], self.splits["y_train"][target_parallel]
            )

            train_pred_parallel = random_forest.predict(self.splits["x_train"])
            acc_train_parallel = metrics.accuracy_score(
                self.splits["y_train"][target_parallel], train_pred_parallel
            )

            y_pred_parallel = random_forest.predict(self.splits["x_test"])
            acc_test_parallel = metrics.accuracy_score(
                self.splits["y_test"][target_parallel], y_pred_parallel
            )
            results = [
                target_parallel,
                params["n_estimators"],
                params["max_depth"],
                params["min_samples_leafs"],
                params["min_samples_splits"],
                acc_train_parallel,
                acc_test_parallel,
                y_pred_parallel.tolist(),
            ]
            results_dict[target_parallel].extend(results)

        return results_dict

    def run_grid_search(
        self,
        n_jobs=-1,
        verbose=1,
        drop_zeros=True,
        best_results=False,
    ) -> pd.DataFrame:
        """
        Perform grid search and return results in a DataFrame.

        Parameters:
        -----------
        n_jobs : int, optional
            Number of parallel jobs to run, by default -1.
        verbose : int, optional
            Verbosity level, by default 1.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the grid search results.
        """
        param_grid = {
            "n_estimators": self.tree_params["n_estimators"],
            "max_depth": self.tree_params["max_depths"],
            "min_samples_leafs": self.tree_params["min_samples_leafs"],
            "min_samples_splits": self.tree_params["min_samples_splits"],
        }

        grid = ParameterGrid(param_grid)

        results = Parallel(n_jobs, verbose=verbose)(
            delayed(self.analyze_result)(params)
            for params in grid
        )

        parameters_columns = [
            "target",
            "n_estimators",
            "max_depths",
            "min_samples_leafs",
            "min_samples_splits",
            "acc_train",
            "acc_test",
            "y_pred",
        ]

        parameters_df = pd.DataFrame(results).melt()[["value"]].explode("value")

        results_df = pd.DataFrame()
        for column in enumerate(parameters_columns):
            results_df[column[1]] = parameters_df.iloc[column[0]::8]

        if drop_zeros:
            results_df = DataHandler(results_df).drop_zero_predictions("y_pred")

        if best_results:
            results_df = DataHandler(results_df).get_best_results("target")

        return results_df


class DataHandler:
    """
    Class for handling data preprocessing tasks.

    Parameters:
    -----------
    dataframe : pd.DataFrame or pd.Series
        The input DataFrame or Series to be processed.

    Attributes:
    -----------
    data_frame : pd.DataFrame
        The processed DataFrame.

    Methods:
    --------
    drop_zero_predictions(column: str) -> pd.Series
        Drop rows where the specified column has all zero values.

    get_splits(target: list | str, features: str | list[str])
    -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Split the DataFrame into training and testing sets.

    get_best_results(target_column: str) -> pd.DataFrame
        Get the rows in the DataFrame with the best accuracy for each
        unique value in the target_column.

    """

    def __init__(self, dataframe: pd.DataFrame | pd.Series) -> None:
        """
        Initialize the DataHandler object.

        Parameters:
        -----------
        dataframe : pd.DataFrame or pd.Series
            The input DataFrame or Series to be processed.

        Returns:
        --------
        None
        """
        self.data_frame = dataframe.copy()

    def drop_zero_predictions(
        self,
        column: str,
    ) -> pd.Series:
        """
        Drop rows where the specified column has all zero values.

        Parameters:
        -----------
        column : str
            The column name in the DataFrame to check for zero values.

        Returns:
        --------
        pd.Series
            The Series with rows dropped where the specified column
            has all zero values.
        """
        def _is_all_zero(list_values: list) -> bool:
            return all(value == 0 for value in list_values)

        if column not in self.data_frame.columns:
            raise ValueError(
                f"Column '{column}' does not exist in the DataFrame."
            )

        mask = self.data_frame[column].apply(_is_all_zero)

        return self.data_frame[~mask]

    def get_splits(
        self,
        target: list | str,
        features: str | list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into training and testing sets.

        Parameters:
        -----------
        target : list or str
            The target column name(s) to use for generating the training
            and testing sets.
        features : str or list of str, optional
            The list of feature column names to use in the DataFrame.
            If None, defaults to ["IRB_Condition", "Signal", "uptrend"].

        Returns
        -------
        tuple of pd.DataFrame
            The tuple containing training data, training target,
            testing data, and testing target.
        """
        end_train_index = int(self.data_frame.shape[0] / 2)

        x_train = self.data_frame.iloc[:end_train_index]
        y_train = pd.DataFrame()
        x_test = self.data_frame.iloc[end_train_index:]
        y_test = pd.DataFrame()

        df_train = x_train.loc[:, features]
        df_test = x_test.loc[:, features]

        for value in enumerate(target):
            y_train[f"target_{value[0]}"] = x_train[value[1]]
            print(y_train.shape)

        for value in enumerate(target):
            y_test[f"target_{value[0]}"] = x_test[value[1]]
            print(y_test.shape)

        return df_train, y_train, df_test, y_test

    def get_best_results(
        self,
        target_column: str,
    ) -> pd.DataFrame:
        """
        Get the rows in the DataFrame with the best accuracy for each
        unique value in the target_column.

        Parameters:
        -----------
        target_column : str
            The column name in the DataFrame containing target values.

        Returns:
        --------
        pd.DataFrame
            The rows with the best accuracy for each unique value in the
            target_column.
        """
        max_acc_targets = [
            (
                self.data_frame
                .query(f"{target_column} == @target")["acc_test"]
                .astype("float64")
                .idxmax(axis=0)
            )
            for target in self.data_frame[target_column].unique()
        ]

        return self.data_frame.loc[max_acc_targets]

    def drop_outlier(
        self,
        target_column: str,
        iqr_scale: float = 1.5,
        upper_quantile: float = 0.75,
        down_quantile: float = 0.25
    ) -> pd.Series:
        """
        Remove outliers from a specific column using the Interquartile
        Range (IQR) method.

        Parameters:
        -----------
        target_column : str
            The name of the column from which outliers will be removed.
        iqr_scale : float, optional
            A scale factor to adjust the range of the IQR.
            (default: 1.5)
        upper_quantile : float, optional
            The quantile value for the upper bound of the IQR range.
            (default: 0.75 (75th percentile))
        down_quantile : float, optional
            The quantile value for the lower bound of the IQR range.
            (default: 0.25 (25th percentile))

        Returns:
        --------
        pd.Series
            A new Series with outliers replaced by the nearest valid
            values within the IQR range.
        """
        outlier_array = self.data_frame[target_column].copy()

        iqr_range = (
            outlier_array.quantile(upper_quantile)
            - outlier_array.quantile(down_quantile)
        ) * iqr_scale

        upper_bound = (
            outlier_array
            .quantile(upper_quantile)
            + iqr_range
        )

        lower_bound = (
            outlier_array
            .quantile(down_quantile)
            - iqr_range
        )

        outlier_array = np.where(
            outlier_array > upper_bound,
            upper_bound,
            outlier_array
        )

        outlier_array = np.where(
            outlier_array < lower_bound,
            lower_bound,
            outlier_array
        )

        return pd.Series(outlier_array)
