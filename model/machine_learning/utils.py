from typing import Literal

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import ParameterGrid, learning_curve
from joblib import Parallel, delayed
import plotly.express as px
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

    def __init__(
        self,
        dataframe: pd.DataFrame | pd.Series | np.ndarray
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
        method: Literal["simple", "ratio"] = "ratio",
        quantiles: list[float] | None = None
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
        method : Literal["simple", "ratio"], optional
            The method used for calculating class proportions. 'simple'
            returns the raw class counts, 'ratio' returns the
            proportions of the target variable within each quantile.
            (default: "ratio")
        quantiles : list of float or None, optional
            The quantile intervals used for splitting the 'column' into
            quantiles. If None, it will use decile (0-10-20-...-90-100)
            quantiles by default.

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the quantile split analysis. Rows
            correspond to quantile intervals based on the specified
            column, columns correspond to unique values of the target
            variable, and the values represent either counts or
            proportions, depending on the chosen method.
        """
        if isinstance(self.data_frame, pd.Series):
            feature = self.data_frame
        else:
            feature = self.data_frame[column]

        if feature.hasnans:
            feature = feature.dropna()

        if quantiles is None:
            quantiles = np.quantile(
                feature,
                np.arange(0, 1.1, 0.1)
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

class ModelHandler:
    """
    A class for handling machine learning model evaluation.

    Parameters:
    -----------
    estimator : object
        The machine learning model to be evaluated.
    X_train : array-like of shape (n_samples, n_features)
        Training input samples.
    y_train : array-like of shape (n_samples,)
        Target values for training.
    X_test : array-like of shape (n_samples, n_features)
        Testing input samples.
    y_test : array-like of shape (n_samples,)
        True target values for testing.

    Attributes:
    -----------
    x_train : array-like of shape (n_samples, n_features)
        Training input samples.
    y_train : array-like of shape (n_samples,)
        Target values for training.
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
        names = pd.Series(self.y_test).sort_values().astype(str).unique()

        confusion_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)
        column_names = "predicted_" + names
        index_names = "real_" + names

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix,
            columns = column_names,
            index = index_names,
        )

        if self._has_predic_proba:
            auc = metrics.roc_auc_score(self.y_test, self.y_pred_probs)
            gini = 2 * auc - 1
            support = self.y_test.shape[0]
            auc_str = f"""         AUC                         {auc:.4f}      {support}
        Gini                         {gini:.4f}      {support}"""
        else:
            auc_str = ""

        return f"""Confusion matrix
--------------------------------------------------------------
{confusion_matrix_df}


Classification reports
--------------------------------------------------------------

{metrics.classification_report(self.y_test, self.y_pred, digits=4)[:-1]}
{auc_str}
"""

class PlotCurve:
    """
    A class for plotting target curves with specified thresholds.

    Parameters:
    -----------
    data_frame : pd.DataFrame
        The DataFrame containing the data to be plotted.

    Attributes:
    -----------
    data_frame : pd.DataFrame
        The DataFrame containing the data.

    Methods:
    --------
    quantile_distribution(
        target: str,
        feature: str,
        middle_line: float = 0.5,
        step: int | float | None = None,
        **kwargs,
    ):
        Plot the quantile distribution of target values by a feature.

    plot_roc_curve(
        fpr: str | np.ndarray | pd.Series,
        tpr: str | np.ndarray | pd.Series
    ) -> plotly.graph_objs._figure.Figure:
        Plot a Receiver Operating Characteristic (ROC) curve.

    """
    def __init__(self, data_frame: pd.DataFrame) -> None:
        """
        Initialize the PlotCurve object.

        Parameters:
        -----------
        data_frame : pd.DataFrame
            The DataFrame containing the data to be plotted.
        """
        self.data_frame = data_frame
        self.data = None

    def __complex_target_curves(
        self,
        column: str | np.ndarray | pd.Series,
        middle_line: int | float = 0.5,
        step: int | float = 0.02,
        **kwargs,
    ):
        """
        Plot the target curve with specified thresholds.

        Parameters:
        -----------
        column : str
            The name of the DataFrame column to plot.
        middle_line : int or float, optional
            The base line or center line value
            (default: 0.5).
        step : int or float, optional
            The step size for upper and lower bounds
            (default: 0.02).

        Returns:
        --------
        plotly.graph_objs._figure.Figure
            A Plotly figure containing the target curve and thresholds.
        """
        if isinstance(column, str):
            data_frame = self.data[column]

            if isinstance(self.data.index, pd.CategoricalIndex):
                data_frame_indexes = pd.Series(self.data.index).astype(str)
            else:
                data_frame_indexes = self.data.index
        else:
            column = None
            data_frame = self.data
            data_frame_indexes = None

        fig = px.line(data_frame, y=column, x=data_frame_indexes)

        self.__hline_range(middle_line, step, fig, kwargs)

        fig.update_layout(kwargs)
        return fig

    def __hline_range(self, middle_line, step, fig, kwargs):
        upper_bound = middle_line + step
        lower_bound = middle_line - step

        kwargs["upper_bound_color"] = kwargs.get("upper_bound_color", "lime")
        kwargs["middle_line_color"] = kwargs.get("middle_line_color", "grey")
        kwargs["lower_bound_color"] = kwargs.get("lower_bound_color", "red")
        kwargs["line_type"] = kwargs.get("line_type", "dash")
        kwargs["col"] = kwargs.get("col", "all")

        fig.add_hline(
            col=kwargs["col"],
            y=upper_bound,
            line_dash=kwargs["line_type"],
            line_color=kwargs["upper_bound_color"],
            annotation_text=f"Valor Y = {upper_bound}"
        )

        fig.add_hline(
            col=kwargs["col"],
            y=middle_line,
            line_dash=kwargs["line_type"],
            line_color=kwargs["middle_line_color"],
            annotation_text="Center line"
        )

        fig.add_hline(
            col=kwargs["col"],
            y=lower_bound,
            line_dash=kwargs["line_type"],
            line_color=kwargs["lower_bound_color"],
            annotation_text=f"Valor Y = {lower_bound}"
        )

        kwargs.pop('upper_bound_color')
        kwargs.pop('middle_line_color')
        kwargs.pop('lower_bound_color')
        kwargs.pop('line_type')
        kwargs.pop('col')

        return fig

    def plot_roc_curve(
        self,
        fpr: str | np.ndarray | pd.Series,
        tpr: str | np.ndarray | pd.Series
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
        if isinstance(fpr, str):
            fpr = self.data_frame[fpr]
        if isinstance(tpr, str):
            tpr = self.data_frame[tpr]

        fig = px.line(
            x=fpr,
            y=tpr,
            title=f"ROC Curve (AUC={metrics.auc(fpr, tpr):.4f})",
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
            width=700,
            height=700,
            template="plotly_dark"
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

    def quantile_distribution(
        self,
        target: str,
        feature: str,
        middle_line: float = 0.5,
        step: float | None = None,
        **kwargs,
    ):
        """
        Plot the quantile distribution of target values by a feature.

        Parameters
        ----------
        target : str
            The target variable to be plotted.
        feature : str
            The feature used for quantile splitting.
        middle_line : float, optional
            The position of the middle line (default: 0.5).
        step : int, float, or None, optional
            The step size for upper and lower bounds
            (default: None).
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
            .quantile_split(target, feature, "ratio")
            .iloc[:, [1]]
        )

        target_name = data.columns[0]
        data["index"] = data.index

        self.data = (
            pd.DataFrame(
                data.to_numpy(),
                columns=["probability", feature])
        )

        self.data[feature] = self.data[feature].astype("str")
        self.data["probability"] = self.data["probability"].astype("float")
        self.data = self.data.set_index(feature)

        title = (
            f"Quantile Distribution of {target}"
            f" (value: {target_name}) by {feature}"
        )

        if step:
            return (
                self.__complex_target_curves(
                    target_name,
                    middle_line,
                    step,
                    **kwargs
                )
                .update_layout(title=title, title_x=0.5)
            )

        return (
            px.line(self.data, y="probability", title=title)
            .add_hline(y=middle_line)
            .update_layout(title_x=0.5)
            .update_layout(**kwargs)
        )

