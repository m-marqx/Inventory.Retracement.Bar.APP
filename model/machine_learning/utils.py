from typing import Literal
import re

import numpy as np
import pandas as pd
import shap
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import ParameterGrid, learning_curve, train_test_split
from joblib import Parallel, delayed
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go

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
            results_df[column[1]] = parameters_df.iloc[column[0] :: 8]

        if drop_zeros:
            results_df = DataHandler(results_df).drop_zero_predictions("y_pred")

        if best_results:
            results_df = DataHandler(results_df).get_best_results("target")

        return results_df


class DataHandler:
    """
    Class for handling data preprocessing tasks.

    Parameters
    ----------
    dataframe : pd.DataFrame or pd.Series
        The input DataFrame or Series to be processed.

    Attributes
    ----------
    data_frame : pd.DataFrame
        The processed DataFrame.

    Methods
    -------
    get_datasets(feature_columns, test_size=0.5, split_size=0.7)
        Splits the data into development and validation datasets.
    drop_zero_predictions(column)
        Drops rows where the specified column has all zero values.
    get_splits(target, features)
        Splits the DataFrame into training and testing sets.
    get_best_results(target_column)
        Gets the rows with the best accuracy for each unique value in
        the target column.
    result_metrics(result_column=None, is_percentage_data=False,
    output_format="DataFrame")
        Calculates result-related statistics like expected return and win rate.
    fill_outlier(column=None, iqr_scale=1.5, upper_quantile=0.75,
    down_quantile=0.25)
        Removes outliers from a specified column using the IQR method.
    quantile_split(target_input, column=None, method="ratio",
    quantiles=None, log_values=False)
        Splits data into quantiles and analyzes the relationship with a target.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame | pd.Series | np.ndarray,
    ) -> None:
        """
        Initialize the DataHandler object.

        Parameters:
        -----------
        dataframe : pd.DataFrame, pd.Series, or np.ndarray
            The input data to be processed. It can be a pandas DataFrame,
            Series, or a numpy array.

        """
        self.data_frame = dataframe.copy()

        if isinstance(dataframe, np.ndarray):
            self.data_frame = pd.Series(dataframe)

    def get_datasets(
        self,
        feature_columns: list,
        test_size: float = 0.5,
        split_size: float = 0.7
    ) -> dict[dict[pd.DataFrame, pd.Series]]:
        """
        Splits the data into development and validation datasets.

        Separates the DataFrame into training and testing sets for
        development, and a separate validation set, based on the
        specified split and test sizes.

        Parameters
        ----------
        feature_columns : list
            List of column names to be used as features.
        test_size : float
            Proportion of the dataset to include in the test split.
            (default: 0.5)
        split_size : float
            Proportion of the dataset to include in the development
            split.
            (default: 0.7)

        Returns
        -------
        dict
            A dictionary containing the development and validation
            datasets, each itself a dictionary with DataFrames and
            Series for features and target values respectively.

        Raises
        ------
        ValueError
            If the provided data_frame is not a Pandas DataFrame.
        """
        if not isinstance(self.data_frame, pd.DataFrame):
            raise ValueError("The dataframe must be a Pandas DataFrame")

        split_index = int(self.data_frame.shape[0] * split_size)
        development_df = self.data_frame.iloc[:split_index].copy()
        validation_df = self.data_frame.iloc[split_index:].copy()

        features = development_df[feature_columns]
        target = development_df["Target_1_bin"]

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            shuffle=False
        )

        origin_datasets = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }
        validation_dataset = {
            "X_validation": validation_df[feature_columns],
            "y_validation": validation_df["Target_1_bin"]
        }

        return {
            "development": origin_datasets,
            "validation": validation_dataset
        }

    def model_pipeline(
        self,
        features_columns: list,
        target_column: str,
        estimator: object,
        return_series: pd.Series,
        validation_size: float = 0.3,
    ) -> pd.DataFrame:
        """
        Execute a machine learning pipeline for model evaluation.

        This method performs a machine learning pipeline, including
        data splitting, training, validation, and evaluation.

        Parameters:
        -----------
        features_columns : list
            List of column names representing features used for training
            the model.
        target_column : str
            Name of the target variable column.
        estimator : object
            Machine learning model (estimator) to be trained and
            evaluated.
        return_series : pd.Series
            Series containing the target variable for the model.
        validation_size : float, optional
            Proportion of the dataset to include in the validation
            split.
            (default: 0.3)

        Returns:
        --------
        pd.DataFrame
            DataFrame containing model returns and validation date.

        Raises:
        -------
        ValueError
            If validation_size is outside the valid range (0.0 to 1.0).
        """
        if validation_size > 1 or validation_size < 0:
            raise ValueError("validation_size should be between 0.0 and 1.0")

        split_size = 1 - validation_size
        split_index = int(self.data_frame.shape[0] * split_size)
        development = self.data_frame.iloc[:split_index].copy()
        validacao = self.data_frame.iloc[split_index:].copy()

        features = development[features_columns]
        target = development[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.5,
            shuffle=False
        )

        estimator.fit(X_train, y_train)

        validation_x_test = validacao[features_columns]
        validation_y_test = validacao[target_column]

        x_series = pd.concat([X_test, validation_x_test], axis=0)
        y_series = pd.concat([y_test, validation_y_test], axis=0)

        model_returns = (
            ModelHandler(estimator, x_series, y_series)
            .model_returns(return_series)
        )
        model_returns['validation_date'] = str(validacao.index[0])
        return model_returns

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

    def result_metrics(
        self,
        result_column: str = None,
        is_percentage_data: bool = False,
        output_format: Literal["dict", "Series", "DataFrame"] = "DataFrame",
    ) -> dict[float, float, float, float] | pd.Series | pd.DataFrame:
        """
        Calculate various statistics related to results, including
        expected return, win rate, positive and negative means, and
        payoff ratio.

        Parameters:
        -----------
        result_column : str, optional
            The name of the column containing the results (returns) for
            analysis.
            If None, the instance's data_frame will be used as the
            result column.
            (default: None).
        is_percentage_data : bool, optional
            Indicates whether the data represents percentages.
            (default: False).
        output_format : Literal["dict", "Series", "DataFrame"],
        optional
            The format of the output. Choose from 'dict', 'Series', or
            'DataFrame'
            (default: 'DataFrame').

        Returns:
        --------
        dict or pd.Series or pd.DataFrame
            Returns the calculated statistics in the specified format:
            - If output_format is `'dict'`, a dictionary with keys:
                - 'Expected_Return': float
                    The expected return based on the provided result
                    column.
                - 'Win_Rate': float
                    The win rate (percentage of positive outcomes) of
                    the model.
                - 'Positive_Mean': float
                    The mean return of positive outcomes from the
                    model.
                - 'Negative_Mean': float
                    The mean return of negative outcomes from the
                    model.
                - 'Payoff': float
                    The payoff ratio, calculated as the positive mean
                    divided by the absolute value of the negative mean.
                - 'Observations': int
                    The total number of observations considered.
            - If output_format is `'Series'`, a pandas Series with
            appropriate index labels.
            - If output_format is `'DataFrame'`, a pandas DataFrame
            with statistics as rows and a 'Stats' column as the index.

        Raises:
        -------
        ValueError
            If output_format is not one of `'dict'`, `'Series'`, or
            `'DataFrame'`.
        ValueError
            If result_column is `None` and the input data_frame is not
            a Series.
        """
        data_frame = self.data_frame.copy()

        if is_percentage_data:
            data_frame = (data_frame - 1) * 100

        if output_format not in ["dict", "Series", "DataFrame"]:
            raise ValueError(
                "output_format must be one of 'dict', 'Series', or "
                "'DataFrame'."
            )

        if result_column is None:
            if isinstance(data_frame, pd.Series):
                positive = data_frame[data_frame > 0]
                negative = data_frame[data_frame < 0]
                positive_mean = positive.mean()
                negative_mean = negative.mean()
            else:
                raise ValueError(
                    "result_column must be provided for DataFrame input."
                )

        else:
            positive = data_frame.query(f"{result_column} > 0")
            negative = data_frame.query(f"{result_column} < 0")
            positive_mean = positive[result_column].mean()
            negative_mean = negative[result_column].mean()

        win_rate = (
            positive.shape[0]
            / (positive.shape[0] + negative.shape[0])
        )

        expected_return = (
            positive_mean
            * win_rate
            - negative_mean
            * (win_rate - 1)
        )

        payoff = positive_mean / abs(negative_mean)

        results = {
            "Expected_Return": expected_return,
            "Win_Rate": win_rate,
            "Positive_Mean": positive_mean,
            "Negative_Mean": negative_mean,
            "Payoff" : payoff,
            "Observations" : positive.shape[0] + negative.shape[0],
        }

        stats_str = "Stats %" if is_percentage_data else "Stats"
        if output_format == "Series":
            return pd.Series(results).rename(stats_str)
        if output_format == "DataFrame":
            return pd.DataFrame(
                results,
                index=["Value"]
            ).T.rename_axis(stats_str)

        return results

    def fill_outlier(
        self,
        column: str = None,
        iqr_scale: float = 1.5,
        upper_quantile: float = 0.75,
        down_quantile: float = 0.25,
    ) -> pd.Series:
        """
        Remove outliers from a given target column using the IQR
        (Interquartile Range) method.

        Parameters:
        -----------
        column : str, optional
            The name of the target column containing the data to be processed.
            If None, the instance's data_frame will be used as the target.
        iqr_scale : float, optional
            The scaling factor to determine the outlier range based on the IQR.
            (default: 1.5)
        upper_quantile : float, optional
            The upper quantile value for calculating the IQR.
            (default: 0.75 (75th percentile))
        down_quantile : float, optional
            The lower quantile value for calculating the IQR.
            (default: 0.25 (25th percentile))

        Returns:
        --------
        pd.Series
            A Series with outliers removed based on the specified
            criteria.
        """
        if column is None:
            if isinstance(self.data_frame, pd.Series):
                outlier_array = self.data_frame.copy()
            else:
                raise ValueError(
                    "column must be provided for DataFrame input."
                )
        else:
            outlier_array = self.data_frame[column].copy()

        iqr_range = (
            outlier_array.quantile(upper_quantile)
            - outlier_array.quantile(down_quantile)
        ) * iqr_scale

        upper_bound = outlier_array.quantile(upper_quantile) + iqr_range

        lower_bound = outlier_array.quantile(down_quantile) - iqr_range

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

        return pd.Series(outlier_array, index=self.data_frame.index)

    def quantile_split(
        self,
        target_input: str | pd.Series | np.ndarray,
        column: str = None,
        method: Literal["simple", "ratio", "sum", "prod"] | None = "ratio",
        quantiles: np.ndarray | pd.Series | int = 10,
        log_values: bool = False,
    ) -> pd.DataFrame:
        """
        Split data into quantiles based on a specified column and
        analyze the relationship between these quantiles and a target
        variable.

        Parameters:
        -----------
        target_input : str, pd.Series, or np.ndarray
            The target variable for the analysis. It can be a column
            name, a pandas Series, or a numpy array.
        column : str, optional
            The name of the column used for quantile splitting.
        method : Literal["simple", "ratio", "sum", "prod"], optional
            The method used for calculating class proportions. 'simple'
            returns the raw class counts, 'ratio' returns the
            proportions of the target variable within each quantile.
            (default: "ratio")
        quantiles : np.ndarray or pd.Series or int, optional
            The quantile intervals used for splitting the 'feature' into
            quantiles. If an integer is provided, it represents the
            number of quantiles to create. If an array or series is
            provided, it specifies the quantile boundaries.
            (default: 10).
        log_values : bool, optional
            If True and 'method' is 'prod', the resulting values are
            computed using logarithmic aggregation.
            (default: False)

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the quantile split analysis. Rows
            correspond to quantile intervals based on the specified
            column, columns correspond to unique values of the target
            variable, and the values represent either counts or
            proportions, depending on the chosen method and split type.
        """
        if method in ["prod", "sum"]:
            split_type = 'data'
        elif method in ["simple","ratio"]:
            split_type = 'frequency'
        else:
            raise ValueError(
                "method must be prod, sum,"
                f" simple or ratio instead of {method}"
            )

        if isinstance(self.data_frame, pd.Series):
            feature = self.data_frame
        else:
            feature = self.data_frame[column]

        if feature.hasnans:
            feature = feature.dropna()

        if isinstance(quantiles, int):
            range_step = 1 / quantiles
            quantiles = np.quantile(
                feature,
                np.arange(0, 1.01, range_step)
            )

            quantiles = np.unique(quantiles)

        if isinstance(target_input, str):
            target_name = target_input
            target = self.data_frame[target_input]
        else:
            target_name = "target"
            target = pd.Series(target_input)

        if feature.index.dtype != target.index.dtype:
            feature = feature.reset_index(drop=True)
            target = target.reset_index(drop=True)

        if not target.index.equals(feature.index):
            target = target.reindex(feature.index)

        class_df = pd.cut(
            feature,
            quantiles,
            include_lowest=True,
        )

        feature_name = column if column else "feature"

        quantile_df = pd.DataFrame(
            {
                feature_name: class_df,
                target_name: target
            }
        )
        if split_type == 'data':
            quantile_df = quantile_df.groupby(feature_name)[target_name]

            if method == 'sum':
                quantile_df = quantile_df.sum()
            if method == 'prod':
                if log_values:
                    quantile_df = np.log(quantile_df.prod())
                else:
                    quantile_df = quantile_df.prod() - 1

        else:
            quantile_df = pd.crosstab(
                index=quantile_df[feature_name],
                columns=quantile_df[target_name],
            )

            if method == "ratio":
                quantile_df = (
                    quantile_df
                    .div(quantile_df.sum(axis=1), axis=0)
                )
        return quantile_df


    def get_split_variable(
        self,
        target_input: str,
        column: str,
        quantiles: np.ndarray | pd.Series | int = 10,
        method: Literal["simple", "ratio", "sum", "prod"] = "ratio",
        log_values: bool = False,
        threshold: float = 0.5,
        higher_than_threshold: bool = True,
    ) -> pd.Series:
        """
        Get a binary variable based on quantile analysis.

        This method performs quantile analysis on the specified column
        using the provided target variable and threshold. It creates a
        binary variable indicating whether the values in the column fall
        within specific quantile intervals.

        Parameters:
        -----------
        target_input : str, pd.Series, or np.ndarray
            The target variable for the analysis. It can be a column
            name, a pandas Series, or a numpy array.
        column : str, optional
            The name of the column used for quantile splitting.
        method : Literal["simple", "ratio", "sum", "prod"], optional
            The method used for calculating class proportions. 'simple'
            returns the raw class counts, 'ratio' returns the
            proportions of the target variable within each quantile.
            (default: "ratio")
        quantiles : np.ndarray or pd.Series or int, optional
            The quantile intervals used for splitting the 'feature' into
            quantiles. If an integer is provided, it represents the
            number of quantiles to create. If an array or series is
            provided, it specifies the quantile boundaries.
            (default: 10).
        log_values : bool, optional
            If True and 'method' is 'prod', the resulting values are
            computed using logarithmic aggregation.
            (default: False)
        threshold : float or int, optional
            The threshold value for determining the quantile intervals.
            Values above this threshold will be considered.
            (default: 0.5)

        Returns:
        --------
        pd.Series
            A binary variable indicating whether the values in the
            specified column fall within the determined quantile
            intervals.
        """
        split_data = self.quantile_split(
            target_input,
            column,
            method,
            quantiles,
            log_values,
        )

        split_data = (
            split_data.iloc[:, 1] if split_data.shape[1] == 2
            else split_data
        )

        if higher_than_threshold:
            data = split_data[split_data > threshold]
        else:
            data = split_data[split_data < threshold]

        intervals = [[x[0].left, x[0].right] for x in data.items()]
        variable = pd.Series(False, index=self.data_frame.index)

        for x in intervals:
            variable |= self.data_frame[column].between(x[0], x[1])

        lower_bound = (
            data.index[0].right if data.iloc[0] > threshold
            else None
        )

        upper_bound = (
            data.index[-1].left if data.iloc[-1] > threshold
            else None
        )

        if upper_bound:
            variable |= self.data_frame[column] > upper_bound

        if lower_bound:
            variable |= self.data_frame[column] <= lower_bound
        return variable

    def get_split_variable_intervals(
        self,
        target_input: str,
        column: str,
        quantiles: np.ndarray | pd.Series | int = 10,
        method: Literal["simple", "ratio", "sum", "prod"] = "ratio",
        log_values: bool = False,
        threshold: float = 0.5,
        higher_than_threshold: bool = True,
    ) -> pd.Series:
        """
        Get intervals from quantile-split variable analysis.

        This method performs quantile analysis on the specified column
        using the provided target variable, method, and threshold. It
        returns intervals based on values higher or lower than the
        specified threshold.

        Parameters:
        -----------
        target_input : str, pd.Series, or np.ndarray
            The target variable for the analysis. It can be a column
            name, a pandas Series, or a numpy array.
        column : str, optional
            The name of the column used for quantile splitting.
        method : Literal["simple", "ratio", "sum", "prod"], optional
            The method used for calculating class proportions. 'simple'
            returns the raw class counts, 'ratio' returns the
            proportions of the target variable within each quantile.
            (default: "ratio")
        quantiles : np.ndarray or pd.Series or int, optional
            The quantile intervals used for splitting the 'feature' into
            quantiles. If an integer is provided, it represents the
            number of quantiles to create. If an array or series is
            provided, it specifies the quantile boundaries.
            (default: 10).
        log_values : bool, optional
            If True and 'method' is 'prod', the resulting values are
            computed using logarithmic aggregation.
            (default: False)
        threshold : float or int, optional
            The threshold value for determining the quantile intervals.
            Values above this threshold will be considered.
            (default: 0.5)
        higher_than_threshold : bool, optional
            If True, values higher than the threshold are considered
            for quantile intervals. If False, values lower than the
            threshold are considered.
            (default: True)

        Returns:
        --------
        pd.Series
            A dictionary containing the intervals based on quantile
            analysis.
        """
        split_data = self.quantile_split(
            target_input,
            column,
            method,
            quantiles,
            log_values,
        )

        split_data = (
            split_data.iloc[:, 1] if split_data.shape[1] == 2
            else split_data
        )

        if higher_than_threshold:
            data = split_data[split_data > threshold]
        else:
            data = split_data[split_data < threshold]

        intervals = [[x[0].left, x[0].right] for x in data.items()]
        variable_intervals = {}

        for x in enumerate(intervals):
            variable_intervals[f'interval_{x[0]}'] = x[1]

        if data.shape[0] > 0:
            lower_bound = (
                data.index[0].right if data.iloc[0] > threshold
                else None
            )

            upper_bound = (
                data.index[-1].left if data.iloc[-1] > threshold
                else None
            )

            variable_intervals['upper_bound'] = upper_bound
            variable_intervals['lower_bound'] = lower_bound

        return variable_intervals

    def get_intervals_variables(
        self,
        column: str,
        intervals: dict
    ) -> pd.Series:
        """
        Get a binary variable based on specified intervals.

        This method creates a binary variable indicating whether the
        values in the specified column fall within the given intervals.

        Parameters:
        -----------
        column : str
            The name of the column to analyze.
        intervals : dict
            A dictionary defining the intervals. The keys represent the
            names of the intervals, and the values can be:
            - A list [start, end] defining a closed interval.
            - A single value for open intervals.

        Returns:
        --------
        pd.Series
            A binary variable indicating whether the values in the
            specified column fall within the specified intervals.
        """
        variable = pd.Series(False, index=self.data_frame.index)
        interval_list = list(intervals.values())

        for x in intervals.values():
            if isinstance(x, list):
                variable |= self.data_frame[column].between(x[0], x[1])

            #This case will be handled by `get_split_variable_intervals` method
            elif x == interval_list[-2] and not isinstance(x, list):
                if x:
                    variable |= self.data_frame[column] > x

            elif x == interval_list[-1] and not isinstance(x, list):
                if x:
                    variable |= self.data_frame[column] <= x
        return variable


class ModelHandler:
    """
    A class for handling machine learning model evaluation.

    Parameters:
    -----------
    estimator : object
        The machine learning model to be evaluated.
    X_test : array-like of shape (n_samples, n_features)
        Testing input samples.
    y_test : array-like of shape (n_samples,)
        True target values for testing.

    Attributes:
    -----------
    x_test : array-like of shape (n_samples, n_features)
        Testing input samples.
    y_test : array-like of shape (n_samples,)
        True target values for testing.
    estimator : object
        The machine learning model.
    y_pred_probs : array-like of shape (n_samples,), optional
        Predicted class probabilities (if available).
    _has_predic_proba : bool
        Indicates whether the estimator has predict_proba method.

    Properties:
    -----------
    results_report : str
        A string containing a results report including a confusion matrix,
        a classification report, AUC, Gini index (if predict_proba is
        available), and support.
    """

    def __init__(self, estimator, X_test, y_test) -> None:
        """
        Initialize the ModelHandler object.

        Parameters:
        -----------
        estimator : object
            An instance of a scikit-learn estimator for classification or
            regression.
        X_test : array-like of shape (n_samples, n_features)
            Test input samples.
        y_test : array-like of shape (n_samples,)
            True target values for testing.
        """
        self.x_test = X_test
        self.y_test = y_test
        self.estimator = estimator
        self.y_pred_probs = None
        self.y_pred = estimator.predict(X_test)
        self._has_predic_proba = (
            hasattr(estimator, 'predict_proba')
            and callable(getattr(estimator, 'predict_proba'))
        )

        if self._has_predic_proba:
            self.y_pred_probs = estimator.predict_proba(X_test)[:, 1]

    def model_returns(
        self,
        return_series: pd.Series,
        fee: float = 0.1,
    ) -> tuple[pd.DataFrame, str]:
        """
        Calculate returns and performance metrics for a trading model.

        This method calculates returns and various performance metrics
        for a trading model using predicted probabilities and actual
        returns. It takes into account transaction fees for trading.

        Parameters:
        -----------
        return_series : pd.Series
            A pandas Series containing the actual returns of the trading
            strategy.
        fee : float, optional
            The transaction fee as a percentage (e.g., 0.1% for 0.1)
            for each trade.
            (default: 0.1)

        Returns:
        --------
        tuple[pd.DataFrame, str]
            A tuple containing:
            - pd.DataFrame: A DataFrame with various columns
            representing the trading results
            - str: A message indicating the success of the operation

        Raises:
        -------
        ValueError:
            If the estimator isn't suitable for classification
            (predict_proba isn't available).
        """
        if not self._has_predic_proba:
            raise ValueError(
                "The estimator isn't suitable for classification"
                " (predict_proba isn't available)."
            )

        if return_series.min() > 0:
            return_series = return_series - 1

        fee = fee / 100
        df_returns = (
            pd.DataFrame(
                {'y_pred_probs' : self.y_pred_probs},
                index=self.x_test.index
            )
        )

        period_return = return_series.reindex(df_returns.index)

        df_returns["Period_Return"] = period_return

        df_returns["Position"] = np.where(
            (df_returns["y_pred_probs"] > 0.5), 1, -1
        )

        df_returns["Result"] = (
            df_returns["Period_Return"]
            * df_returns["Position"]
            + 1
        )

        df_returns["Liquid_Result"] = np.where(
            (df_returns["Position"] != 0)
            & (df_returns["Result"].abs() != 1),
            df_returns["Result"] - fee, 1
        )

        df_returns["Period_Return_cum"] = (
            df_returns["Period_Return"] + 1
        ).cumprod()

        df_returns["Total_Return"] = df_returns["Result"].cumprod()
        df_returns["Liquid_Return"] = df_returns["Liquid_Result"].cumprod()

        df_returns["Period_Return_cum_log"] = np.log(
            df_returns["Period_Return_cum"]
        )

        df_returns["Total_Return_log"] = np.log(
            df_returns["Total_Return"]
        )

        df_returns["Liquid_Return_log"] = np.log(
            df_returns["Liquid_Return"]
        )

        df_returns["max_Liquid_Return"] = (
            df_returns["Liquid_Return"].expanding(365).max()
        )

        df_returns["max_Liquid_Return"] = np.where(
            df_returns["max_Liquid_Return"].diff(),
            np.nan, df_returns["max_Liquid_Return"],
        )

        df_returns["drawdown"] = (
            1 - df_returns["Liquid_Return"]
            / df_returns["max_Liquid_Return"]
        )

        drawdown_positive = df_returns["drawdown"] > 0

        df_returns["drawdown_duration"] = drawdown_positive.groupby(
            (~drawdown_positive).cumsum()
        ).cumsum()


        df_returns["max_Liquid_Return_log"] = (
            df_returns["Liquid_Return_log"].expanding(365).max()
        )

        df_returns["max_Liquid_Return_log"] = np.where(
            df_returns["max_Liquid_Return_log"].diff(),
            np.nan, df_returns["max_Liquid_Return_log"],
        )

        df_returns["drawdown_log"] = (
            1 - df_returns["Liquid_Return_log"]
            / df_returns["max_Liquid_Return_log"]
        )

        drawdown_positive_log = df_returns["drawdown"] > 0

        df_returns["drawdown_duration_log"] = drawdown_positive_log.groupby(
            (~drawdown_positive_log).cumsum()
        ).cumsum()
        return df_returns

    def shapley_values(
        self,
        output: Literal["DataFrame", "Figure"] = "Figure",
        **kwargs,
    ) -> go.Figure:
        """
        Calculate and visualize Shapley values for feature importance.

        This method calculates the Shapley values for feature
        importanceusing the SHAP (SHapley Additive exPlanations)
        library. It then visualizes the Shapley values as a bar chart
        to show the impact of each feature on the model's predictions.

        Parameters:
        -----------
        output : Literal["DataFrame", "Figure"], optional
            The output format for the Shapley values. Choose between
            "DataFrame" to return the results as a Pandas DataFrame or
            "Figure" to generate a Plotly bar chart
            (default: "Figure").

        Returns:
        --------
        pd.DataFrame or plotly.graph_objs._figure.Figure
            If `output` is set to "DataFrame," it returns a Pandas
            DataFrame containing the feature names and their
            corresponding Shapley values. If `output` is set to
            "Figure," it returns a Plotly bar chart displaying the
            Shapley values for each feature, where the length of the
            bars represents the magnitude of impact on the model's
            predictions.
        """
        shap_values = (
            shap
            .Explainer(self.estimator)
            .shap_values(self.x_test)
        )

        # Calcular o valor médio SHAP para cada recurso
        mean_shap_values = abs(shap_values).mean(axis=0)

        # Criar um DataFrame para facilitar a manipulação dos dados
        shap_df = pd.DataFrame(
            {
                "Feature": self.x_test.columns,
                "Shapley_values": mean_shap_values
            }
        )

        shap_df = shap_df.sort_values(by="Shapley_values", ascending=True)
        if output == "DataFrame":
            return shap_df

        return px.bar(
            shap_df,
            y='Feature',
            x="Shapley_values",
            **kwargs
        )

    def roc_curve(
        self,
        output: Literal["DataFrame", "Figure"] = "Figure",
    ):
        """
        Plot a Receiver Operating Characteristic (ROC) curve.

        The ROC curve is a graphical representation of the classifier's
        ability to distinguish between positive and negative classes.
        It is created by plotting the True Positive Rate (TPR) against
        the False Positive Rate (FPR) at various threshold settings.

        Parameters:
        -----------
        fpr : str, np.ndarray, or pd.Series
            An array containing the False Positive Rates for different
            classification thresholds.
        tpr : str, np.ndarray, or pd.Series
            An array containing the True Positive Rates for different
            classification thresholds.

        Returns:
        --------
        plotly.graph_objs._figure.Figure
            A Plotly figure displaying the ROC curve with AUC
            (Area Under the Curve) score.

        """
        if output not in ["DataFrame", "Figure"]:
            raise ValueError("output must be 'DataFrame' or 'Figure'")

        fpr, tpr, thresholds = (
            metrics.roc_curve(self.y_test, self.y_pred_probs)
        )

        roc_curve = pd.DataFrame(
            {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
            }
        )

        if output == "Figure":
            roc_auc = metrics.auc(fpr, tpr)

            fig = px.line(
                roc_curve,
                x=fpr,
                y=tpr,
                title=f"ROC Curve (AUC={roc_auc:.4f})",
                labels=dict(x="False Positive Rate", y="True Positive Rate"),
                width=700,
                height=700,
            )

            fig.add_shape(
                type="line",
                line=dict(dash="dash"),
                x0=0,
                x1=1,
                y0=0,
                y1=1,
                opacity=0.65,
            )
            return fig

        return roc_curve

    def learning_curve(
        self,
        train_size: np.ndarray | pd.Series = None,
        k_fold: int = 5
    ) -> pd.DataFrame:
        """
        Generate a learning curve for the estimator.

        A learning curve shows the training and testing scores of an
        estimator
        for varying numbers of training samples. This can be useful to
        evaluate
        how well the estimator performs as more data is used for
        training.

        Parameters:
        -----------
        train_size : np.ndarray or pd.Series, optional
            An array of training sizes or a Series of proportions to
            use for plotting the learning curve. If None, it defaults
            to evenly spaced values between 0.1 and 1.0
            (default: None).
        k_fold : int, optional
            The number of cross-validation folds to use for computing
            scores
            (default: 5).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the mean and standard deviation of
            train and test scores for different training sizes. Columns
            include:
            - 'train_mean': Mean training score
            - 'train_std': Standard deviation of training score
            - 'test_mean': Mean test score
            - 'test_std': Standard deviation of test score

        Notes:
        ------
        The learning curve is generated using cross-validation to
        compute scores.
        """
        if train_size is None:
            train_size = np.linspace(0.1, 1.0, 20)

        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.estimator,
            X=self.x_test,
            y=self.y_test,
            train_sizes=train_size,
            cv=k_fold,
        )

        train_scores_df = pd.DataFrame(train_scores, index=train_sizes)
        train_scores_concat = pd.concat(
            [
                train_scores_df.mean(axis=1),
                train_scores_df.std(axis=1)
            ], axis=1).rename(columns={0: 'train_mean', 1: 'train_std'})

        test_scores_df = pd.DataFrame(test_scores, index=train_sizes)

        test_scores_concat = pd.concat(
            [
                test_scores_df.mean(axis=1),
                test_scores_df.std(axis=1)
            ], axis=1).rename(columns={0: 'test_mean', 1: 'test_std'})

        return pd.concat(
            [
                train_scores_concat,
                test_scores_concat,
            ], axis=1)

    @property
    def results_report(self) -> str:
        """
        Generate a results report including a confusion matrix and a
        classification report.

        Returns:
        --------
        str
            A string containing the results report.
        """
        if not self._has_predic_proba:
            raise ValueError(
                "The estimator isn't suitable for classification"
                " (predict_proba isn't available)."
            )

        names = pd.Series(self.y_test).sort_values().astype(str).unique()

        confusion_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)
        column_names = "predicted_" + names
        index_names = "real_" + names

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix,
            columns=column_names,
            index=index_names,
        )

        auc = metrics.roc_auc_score(self.y_test, self.y_pred_probs)
        gini = 2 * auc - 1
        support = self.y_test.shape[0]
        classification_report = metrics.classification_report(
            self.y_test, self.y_pred, digits=4
        )[:-1]

        auc_str = (
            f"\n         AUC                         {auc:.4f}"
            f"      {support}"
            f"\n        Gini                         {gini:.4f}"
            f"      {support}"
        )

        confusion_matrix_str = (
            f"Confusion matrix"
            f"\n--------------------------------------------------------------"
            f"\n{confusion_matrix_df}"
            f"\n"
            f"\n"
            f"\nClassification reports"
            f"\n--------------------------------------------------------------"
            f"\n"
            f"\n{classification_report}"
            f"{auc_str}"
        )
        return confusion_matrix_str

class DataCurve:
    """
    A class for plotting target curves with specified thresholds.

    Parameters:
    -----------
    data_frame : pd.DataFrame
        The DataFrame containing the data to be plotted.

    target : str
        The target variable to be used in quantile distribution.

    feature : str
        The feature used for quantile splitting.

    quantiles : list of float or None, optional
        The quantile intervals used for splitting the 'feature' into
        quantiles. If None, it will use decile (0-10-20-...-90-100)
        quantiles by default.

    Attributes:
    -----------
    data_frame : pd.DataFrame
        The DataFrame containing the data.

    target : str
        The target variable used in quantile distribution.

    feature : str
        The feature used for quantile splitting.

    quantiles : np.ndarray | pd.Series
        The quantile intervals used for splitting.

    Methods:
    --------
    quantile_distribution(
        middle_line: float = 0.5,
        step: float | None = None,
        show_histogram: bool = False,
        **kwargs,
    ) -> plotly.graph_objs.Figure:
        Plot the quantile distribution of target values by a feature.

    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        target: str,
        feature: str,
        quantiles: np.ndarray | pd.Series | int = 10,
        middle_line: float = 0.5,
        step: float | None = None,
    ) -> None:
        """
        Initialize the DataCurve object.

        Parameters:
        -----------
        data_frame : pd.DataFrame
            The DataFrame containing the data to be plotted.
        target : str
            The target variable to be plotted.
        feature : str
            The feature used for quantile splitting.
        quantiles : np.ndarray or pd.Series or int, optional
            The quantile intervals used for splitting the 'feature' into
            quantiles. If an integer is provided, it represents the
            number of quantiles to create. If an array or series is
            provided, it specifies the quantile boundaries.
            (default: 10).
        middle_line : int or float, optional
            The base line or center line value
            (default: 0.5).
        step : int or float, optional
            The step size for upper and lower bounds
            (default: 0.02).
        """
        self.data_frame = data_frame
        self.data = None
        self.target = target
        self.feature = feature
        self.middle_line = middle_line
        self.step = step
        self.quantiles = quantiles

    def _hline_range(self, fig, kwargs):
        upper_bound = self.middle_line + self.step
        lower_bound = self.middle_line - self.step

        kwargs["upper_bound_color"] = kwargs.get("upper_bound_color", "lime")
        kwargs["middle_line_color"] = kwargs.get("middle_line_color", None)
        kwargs["lower_bound_color"] = kwargs.get("lower_bound_color", "red")
        kwargs["line_type"] = kwargs.get("line_type", "dash")
        kwargs["col"] = kwargs.get("col", "all")
        kwargs["row"] = kwargs.get("row", "all")

        fig.add_hline(
            col=kwargs["col"],
            row=kwargs["row"],
            y=upper_bound,
            line_dash=kwargs["line_type"],
            line_color=kwargs["upper_bound_color"],
            annotation_text=f"upper line: {upper_bound}",
        )

        fig.add_hline(
            col=kwargs["col"],
            row=kwargs["row"],
            y=self.middle_line,
            line_dash=kwargs["line_type"],
            line_color=kwargs["middle_line_color"],
            annotation_text="Center line",
        )

        fig.add_hline(
            col=kwargs["col"],
            row=kwargs["row"],
            y=lower_bound,
            line_dash=kwargs["line_type"],
            line_color=kwargs["lower_bound_color"],
            annotation_text=f"lower value: {lower_bound}",
        )

        kwargs.pop("upper_bound_color")
        kwargs.pop("middle_line_color")
        kwargs.pop("lower_bound_color")
        kwargs.pop("line_type")
        kwargs.pop("col")
        kwargs.pop("row")

        return fig

    def quantile_distribution(
        self,
        show_histogram: bool = False,
        lower_limit: float | None = None,
        upper_limit: float | None = None,
        method: Literal['simple', 'ratio', 'sum', 'prod'] = 'ratio',
        plot_type: Literal['line', 'bar'] = 'line',
        **kwargs,
    ):
        """
        Plot the quantile distribution of target values by a feature.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for customizing the
            visualization of the Plotly layout.

            Custom kwargs used when `step` is not `None`:

            - upper_bound_color: str, optional
                Color for the upper threshold line.
            - middle_line_color: str, optional
                Color for the middle line.
            - lower_bound_color: str, optional
                Color for the lower threshold line.
            - line_type: str, optional
                Type of line (e.g., 'solid', 'dash') for threshold lines.

        Returns
        -------
        plotly.graph_objs.Figure
            A Plotly figure displaying the quantile distribution of the
            data.
        """
        data = (
            DataHandler(self.data_frame)
            .quantile_split(
                self.target,
                self.feature,
                method,
                self.quantiles,
                True
            )
        )

        if isinstance(data, pd.Series):
            data = data.rename(self.feature)
            data = pd.DataFrame(data)
        else:
            data = data.iloc[:, [1]]

        target_name = data.columns[0]
        data["index"] = data.index
        data_type = (
            "Aggregated data" if method in ['prod', 'sum']
            else "Probability"
        )

        self.data = (
            pd.DataFrame(
                data.to_numpy(),
                columns=[data_type, self.feature])
        )

        self.data[self.feature] = self.data[self.feature].astype("str")
        self.data[data_type] = self.data[data_type].astype("float")
        self.data = self.data.set_index(self.feature)

        value_info = (
            f"(value: {target_name})" if method in ['simple', 'ratio']
            else ""
        )

        title = (
            f"Quantile Distribution of {self.target}"
            f" {value_info} by {self.feature}"
        )

        if plot_type == 'line':
            plot = px.line
        else:
            plot = px.bar

        if self.step:
            data_frame = self.data[data_type]
            target_curve_fig = plot(data_frame)
            self._hline_range(target_curve_fig, kwargs)
        else:
            target_curve_fig = (
                plot(self.data, y=data_type)
                .add_hline(y=self.middle_line)
            )

        if show_histogram:
            column = 2

            fig = sp.make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Distribution", "Quantile Distribution")
            )

            histogram_fig = px.histogram(self.data_frame, x=self.feature)
            fig.add_trace(histogram_fig.data[0], row=1, col=1)
            fig.add_trace(target_curve_fig.data[0], row=1, col=2)

            if self.step:
                kwargs["col"] = column
                self._hline_range(fig, kwargs)
            else:
                fig.add_hline(col=column, y=self.middle_line)

        else:
            column = 1
            fig = target_curve_fig

        fig = (
            fig.update_layout(title=title, title_x=0.5)
            .update_layout(**kwargs)
        )

        if lower_limit and upper_limit:
            limit_ranges = self.get_limit_ranges(
                lower_limit, upper_limit, fig.data[column - 1]
            )

            lower_range = limit_ranges["lower_range"]["range"]
            higher_range = limit_ranges["higher_range"]["range"]

            fig.add_vline(x=lower_range, col=column, line_color="lime")
            fig.add_vline(x=higher_range, col=column, line_color="red")

            if show_histogram:
                fig.add_vline(x=lower_limit, col=1, line_color="lime")
                fig.add_vline(x=upper_limit, col=1, line_color="red")

        return fig

    def get_limit_ranges(
        self,
        lower_limit: int | float,
        upper_limit: int | float,
        fig_data: go._scatter.Scatter,
    ) -> pd.DataFrame:
        """
        Get limit ranges for buy and sell zones from a Plotly figure.

        This function extracts limit ranges for buy and sell zones
        from a Plotly figure containing data points. It searches for
        data points on the x-axis of the figure that match the specified
        buy and sell zones and returns them as a DataFrame.

        Parameters:
        -----------
        lower_limit : float
            The buy zone value to search for within the figure.
        upper_limit : float
            The sell zone value to search for within the figure.
        fig_data : go._scatter.Scatter
            A Plotly Scattergl object containing data points.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing limit ranges for buy and sell zones.
            The DataFrame has two columns: 'value' and 'range', where
            'value' represents the buy or sell zone value, and 'range'
            represents the corresponding range found in the figure.

        Raises:
        -------
        ValueError
            If 'fig_data' is not a Plotly Scattergl object.
        """
        if isinstance(fig_data, go.Figure):
            raise ValueError("fig_data must be a Plotly Figure data.")

        def limit_ranges_values(str_interval: str):
            match = re.match(r"\((-?\d+\.\d+), (-?\d+\.\d+)\]", str_interval)
            if match:
                lower_value = float(match.group(1))
                higher_value = float(match.group(2))
                return lower_value, higher_value
            return None, None

        ranges_dict = {}
        for element in fig_data["x"]:
            lower_value, higher_value = limit_ranges_values(element)
            if lower_value is not None and higher_value is not None:
                if lower_value <= lower_limit <= higher_value:
                    ranges_dict["lower_range"] = [lower_limit, element]
                if lower_value <= upper_limit <= higher_value:
                    ranges_dict["higher_range"] = [upper_limit, element]
        return pd.DataFrame(ranges_dict, index=["value", "range"])

    def target_distribution(
        self,
        data_target: 'str',
        data_step: float | None = None,
        data_plot_type: Literal['line', 'bar'] = 'line',
        lower_limit: float | None = None,
        upper_limit: float | None = None,
        method: Literal['sum', 'prod'] = 'sum',
        orientation: Literal['vertical', 'horizontal'] = 'horizontal',
        **kwargs,
    ):
        """
        Plot the target distribution and accumulate data by thresholds.

        This method creates a Plotly figure to visualize the target
        distribution and accumulate data within specified thresholds.
        It shows the distribution of target values and the accumulated
        data values within the given thresholds.

        Parameters:
        -----------
        data_target : str
            The name of the target variable for the target distribution.
        data_step : float
            The step size for upper and lower bounds of the thresholds.
        lower_limit : float, optional
            The lower threshold limit for accumulation
            (default: None).
        upper_limit : float, optional
            The upper threshold limit for accumulation
            (default: None).
        method : Literal['sum', 'prod'], optional
            The method used for accumulating data values. 'sum' for sum
            and 'prod' for product
            (default: 'sum').
        orientation: Literal['vertical', 'horizontal'], optional
            The orientation of the subplot. If 'vertical', the
            frequency plot and the accumulated data plot will be
            stacked vertically. If 'horizontal', they will be placed
            side by side.
        **kwargs : dict
            Additional keyword arguments for customizing the Plotly layout.

        Returns:
        --------
        plotly.graph_objs.Figure
            A Plotly figure displaying the target distribution and
            accumulated data within specified thresholds. It includes
            both the frequency plot and the accumulated data plot.

        Raises:
        -------
        ValueError:
            If 'lower_limit' is specified without 'upper_limit', or
            vice versa.
        """
        data_curve = DataCurve(
            self.data_frame,
            data_target,
            self.feature,
            self.quantiles,
            0,
            data_step,
        )

        title = f"Target Distribution of {self.feature}"

        common_params = dict(
            show_histogram=False,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

        frequency_fig = self.quantile_distribution(
        method = 'ratio',
        **common_params,
        )

        data_fig = data_curve.quantile_distribution(
        method = method,
        plot_type = data_plot_type,
        **common_params,
        )

        if orientation == 'vertical':
            subplot_row = 2
            subplot_col = 1
            sub_titles = None
        else:
            subplot_row = 1
            subplot_col = 2
            sub_titles = ("Frequency", "Aggregated Data")

        fig = sp.make_subplots(
            rows=subplot_row,
            cols=subplot_col,
            subplot_titles=sub_titles,
        )

        fig.add_trace(frequency_fig.data[0], row=1, col=1)
        fig.add_trace(data_fig.data[0], row=subplot_row, col=subplot_col)

        fig = (
            fig.update_layout(title=title, title_x=0.5)
            .update_layout(**kwargs)
        )

        if lower_limit and upper_limit:
            limit_ranges = self.get_limit_ranges(
                lower_limit, upper_limit, fig.data[0]
            )

            lower_range = limit_ranges["lower_range"]["range"]
            higher_range = limit_ranges["higher_range"]["range"]

            fig.add_vline(x=lower_range, line_color="lime")
            fig.add_vline(x=higher_range, line_color="red")

        if self.step:
            kwargs["col"] = 1
            kwargs["row"] = 1
            self._hline_range(fig, kwargs)
        else:
            fig.add_hline(row=1, col=1, y=self.middle_line)

        if data_curve.step:
            kwargs["col"] = subplot_col
            kwargs["row"] = subplot_row
            data_curve._hline_range(fig, kwargs)
        else:
            fig.add_hline(row=subplot_row, col=subplot_col, y=0)
        return fig
