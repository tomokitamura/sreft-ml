import math
import subprocess
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import tensorflow as tf
from sklearn.linear_model import LinearRegression


def n2mfrow(n_plots: int, ncol_max: int = 4) -> tuple[int, int]:
    """
    Determines the number of rows and columns required to plot a given number of subplots.

    Args:
        n_plots (int): Total number of subplots.
        ncol_max (int, optional): Maximum number of columns for subplots. Defaults to 4.

    Returns:
        tuple: (number of rows, number of columns)"""
    n_plots = int(n_plots)
    nrow = math.ceil(n_plots / ncol_max)
    ncol = math.ceil(n_plots / nrow)
    return nrow, ncol


def linear_regression_each_subject(
    df: pd.DataFrame, y_columns: list[str]
) -> pd.DataFrame:
    """
    Perform linear regression for each subject (ID) in the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data for the regression. It must include columns for 'ID', 'TIME', and the target variables specified in 'y_columns'.
        y_columns (list[str]): A list of column names (strings) representing the target variables to be regressed.

    Returns:
        pd.DataFrame: A DataFrame with the regression results for each subject.
    """
    model = LinearRegression()
    results = {"ID": df.ID.unique()}

    for y in y_columns:
        slopes = []
        intercepts = []

        for _, group in df.groupby("ID"):
            x_values = group["TIME"].values.reshape(-1, 1)
            y_values = group[y].values

            valid_mask = ~np.isnan(y_values)
            valid_sample_count = valid_mask.sum()

            if valid_sample_count == 0:
                slopes.append(np.nan)
                intercepts.append(np.nan)
                continue

            model.fit(x_values[valid_mask], y_values[valid_mask])

            if valid_sample_count == 1:
                slopes.append(np.nan)
            else:
                slopes.append(model.coef_[0])
            intercepts.append(model.intercept_)

        results[f"{y}_slope"] = slopes
        results[f"{y}_intercept"] = intercepts

    result = pd.DataFrame(results)
    result = result[
        ["ID"] + [i + j for j in ["_slope", "_intercept"] for i in y_columns]
    ]

    return result


def mixed_effect_linear_regression(
    df: pd.DataFrame, y_columns: list[str]
) -> tuple[pd.DataFrame, list]:
    """
    Perform mixed-effects linear regression on the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data for the regression.
            It must include columns for 'ID', 'TIME', and the target variables specified in 'y_columns'.
        y_columns (list[str]): A list of column names (strings) representing the target variables to be regressed.

    Returns:
        tuple: A tuple containing two elements:
            - result (pd.DataFrame): The DataFrame with the fitted regression parameters for each individual.
            - models (list): A list of fitted mixed-effects regression models for each target variable.
    """
    result = pd.DataFrame(df.ID.unique()).set_axis(["ID"], axis=1)
    models = []

    for y in y_columns:
        df_ = (
            df[["ID", "TIME", y]]
            .dropna()
            .reset_index(drop=True)
            .set_axis(["ID", "TIME", "TARGET"], axis=1)
        )
        full_model = smf.mixedlm(
            "TARGET ~ TIME", data=df_, groups="ID", re_formula="~TIME"
        ).fit()
        random_effects = pd.DataFrame(full_model.random_effects).T.values
        params_pop = full_model.params[0:2].values.T
        params_ind = pd.DataFrame(params_pop + random_effects).set_axis(
            [f"{y}_intercept", f"{y}_slope"], axis=1
        )
        params_ind["ID"] = pd.DataFrame(full_model.random_effects).T.index.values
        result = result.merge(params_ind, how="outer")
        models.append(full_model)

    result = result[
        ["ID"] + [i + j for j in ["_slope", "_intercept"] for i in y_columns]
    ]

    return result, models


def split_data_for_sreftml(
    df: pd.DataFrame,
    name_biomarkers: list[str],
    name_covariates: list[str],
    isMixedlm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data for sreftml.

    Args:
        df (pd.DataFrame): Input DataFrame.
        name_biomarkers (list[str]): List of biomarker names.
        name_covariates (list[str]): List of covariate names.
        isMixedlm (bool): Select whether to use a mixed-effects model when computing model_1 features. Default to True.

    Returns:
        tuple: A tuple containing the following arrays:
            - x (pd.DataFrame): Time values.
            - cov (pd.DataFrame): Covariate values.
            - m (pd.DataFrame): Slope and intercept from regression by biomarker.
            - y (pd.DataFrame): Biomarker values.
    """
    df_ = df.copy()
    if len(name_covariates) > 0 and pd.isna(df[name_covariates]).any().any():
        warnings.warn("Missing value imputation was performed for some covariates.")
        df_[name_covariates] = df_[name_covariates].fillna(
            df.loc[:, name_covariates].mean()
        )

    if isMixedlm:
        linreg, models = mixed_effect_linear_regression(df_, name_biomarkers)
        if pd.isna(linreg).any().any():
            warnings.warn("Missing value imputation was performed for some features.")
            prms = [i.params[0] for i in models] + [i.params[1] for i in models]
            labels = [i + j for j in ["_intercept", "_slope"] for i in name_biomarkers]
            dict_slope = dict(zip(labels, prms))
            linreg = linreg.fillna(dict_slope)
    else:
        linreg = linear_regression_each_subject(df_, name_biomarkers)
        if pd.isna(linreg).any().any():
            warnings.warn("Missing value imputation was performed for some features.")
            linreg = linreg.fillna(linreg.mean())

    df_ = df_.merge(linreg)

    x = df_.TIME
    cov = df_[name_covariates]
    m = df_.loc[:, df_.columns.str.contains("_slope|_intercept")]
    y = df_[name_biomarkers]

    return x, cov, m, y


def np_compute_negative_log_likelihood(
    y_true: np.ndarray, y_pred: np.ndarray, lnvar_y: np.ndarray
) -> np.ndarray:
    """
    Computes the negative log likelihood between true and predicted values using numpy.

    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted target values.
        lnvar_y (np.array): Natural logarithm of the variance.

    Returns:
        np.array: The negative log likelihood for each instance.
    """
    neg_ll = lnvar_y + np.power(y_true - y_pred, 2) / np.exp(lnvar_y)
    return np.nansum(neg_ll, axis=1)


def tf_compute_negative_log_likelihood(
    y_true: np.ndarray, y_pred: np.ndarray, lnvar_y: tf.Variable
) -> tf.Tensor:
    """
    Computes the negative log likelihood between true and predicted values using tensorflow.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        lnvar_y (tf.Variable): Natural logarithm of the variance.

    Returns:
        tf.Tensor: The negative log likelihood for each instance.
    """
    is_nan = tf.math.is_nan(y_true)
    y_true = tf.where(is_nan, tf.zeros_like(y_true), y_true)
    y_pred = tf.where(is_nan, tf.zeros_like(y_pred), y_pred)
    neg_ll = lnvar_y + tf.pow(y_true - y_pred, 2) / tf.exp(lnvar_y)
    neg_ll = tf.where(is_nan, tf.zeros_like(neg_ll), neg_ll)

    return tf.reduce_sum(neg_ll, axis=1)


class DummyTransformer:
    def __init__(
        self,
    ):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return X


def get_current_commit_hash() -> str:
    """
    Retrieves the current commit hash of the git repository.

    Returns:
        str: The current commit hash or a placeholder string if an error occurs.
    """
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
        return commit_hash.strip().decode("utf-8")
    except subprocess.CalledProcessError:
        warnings.warn("Could not get the current commit hash.", UserWarning)
        return "commit_hash_not_available"


def clean_duplicate(
    df: pd.DataFrame, cols: list[str], duplicate_key: list[str] | str | None
) -> pd.DataFrame:
    """
    Checks for duplicate entries in the DataFrame based on the specified columns and removes NaNs; also removes duplicate entries if a subset is specified.

    Parameters:
        df (pd.DataFrame): The DataFrame to check and drop duplicates from.
        cols (list[str]): List of column names to check (and remove) for duplicates.
        duplicate_key (list[str] | str | None): If specify, duplicate deletion will be performed. Then, check duplicate within sepecified columns.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed. It includes only the columns specified in cols and duplicate_key.

    Warnings:
        If any duplicates are found in the DataFrame after cleaning, a warning message is displayed.
        The warning message depends on the `subset` parameter:
        - If `subset` is None, the warning message indicates that some records are duplicates across all columns in `cols`.
        - If `subset` is not None, the warning message indicates that some records are duplicates within the same subset.
    """
    if type(duplicate_key) is str:
        duplicate_key = [duplicate_key]

    if duplicate_key is None:
        df_ = df[cols].dropna()
        if df_.duplicated().any():
            warnings.warn(
                "Some records are duplicates. Set duplicate_key if necessary."
            )
    else:
        df_ = df[cols + duplicate_key].dropna().drop_duplicates()
        if df_.duplicated(subset=duplicate_key).any():
            warnings.warn(
                "Duplicate records remain in some duplicate_keys. Add duplicate_key if necessary."
            )

    return df_


def compute_permutation_importance(
    random_seed: int,
    sreft: tf.keras.Model,
    x_test: np.ndarray,
    cov_test: np.ndarray,
    m_test: np.ndarray,
    y_test: np.ndarray,
    n_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute permutation importance of the model.

    Args:
        random_seed (int): The seed for the random number generator.
        sreft (tf.keras.Model): The model for which to calculate permutation importance.
        x_test (np.ndarray): The x test data.
        cov_test (np.ndarray): The covariates test data.
        m_test (np.ndarray): The m test data.
        y_test (np.ndarray): The y test data.
        n_sample (int): The number of samples.

    Returns:
        tuple[np.ndarray, np.ndarray]: The mean and standard deviation of the permutation importance.
    """
    rng = np.random.default_rng(random_seed)
    y_pred = sreft((x_test, cov_test, m_test, y_test)).numpy()
    neglls_orig = np_compute_negative_log_likelihood(y_test, y_pred, sreft.lnvar_y)

    mean_pi = []
    std_pi = []
    n_pi = m_test.shape[1] + cov_test.shape[1]

    for i in range(n_pi):
        pis = []
        for j in range(n_sample):
            if i < m_test.shape[1]:
                m_test_rand = np.copy(m_test)
                rng.shuffle(m_test_rand[:, i])
                y_pred_rand = sreft((x_test, cov_test, m_test_rand, y_test)).numpy()
            else:
                cov_test_rand = np.copy(cov_test)
                rng.shuffle(cov_test_rand[:, i - m_test.shape[1]])
                y_pred_rand = sreft((x_test, cov_test_rand, m_test, y_test)).numpy()

            neglls_rand = np_compute_negative_log_likelihood(
                y_test, y_pred_rand, sreft.lnvar_y
            )
            nglls_diff = neglls_rand - neglls_orig
            temp_pi = np.nanmean(nglls_diff)
            pis.append(temp_pi)

        mean_pi.append(np.mean(pis))
        std_pi.append(np.std(pis))

    return np.array(mean_pi), np.array(std_pi)
