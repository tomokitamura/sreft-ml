import math
import pickle
import subprocess
import warnings

import autograd.numpy as agnp
import lifelines
import numpy as np
import pandas as pd
import shap
import sklearn.preprocessing as sp
import statsmodels.formula.api as smf
import tensorflow as tf
from sklearn.linear_model import LinearRegression


class NullModel:
    def __init__(self, Intercept, TIME):
        self.params = [Intercept, TIME]


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
        if df_["TIME"].nunique() == 1:
            warnings.warn(
                f"Only one time point is available for {y}. The slope cannot be calculated."
            )
            tmp = pd.DataFrame(
                {
                    "ID": df_.ID.unique(),
                    f"{y}_slope": np.nan,
                    f"{y}_intercept": df_.groupby("ID")["TARGET"].mean().values,
                }
            )
            result = result.merge(tmp, how="outer")
            models.append(NullModel(df_.groupby("ID")["TARGET"].mean().mean(), np.nan))
            continue

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
    m = df_.loc[:, df_.columns.str.contains("_slope|_intercept")].dropna(
        axis=1, how="all"
    )
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


def compute_permutation_importance_(
    random_seed: int,
    sreft: tf.keras.Model,
    x_test: np.ndarray,
    cov_test: np.ndarray,
    m_test: np.ndarray,
    y_test: np.ndarray,
    n_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    [Superseded] Compute permutation importance of the model.

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


def compute_permutation_importance(
    random_seed: int,
    sreft: tf.keras.Model,
    cov_test: np.ndarray,
    m_test: np.ndarray,
    n_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute permutation importance of the model.

    Args:
        random_seed (int): The seed for the random number generator.
        sreft (tf.keras.Model): The model for which to calculate permutation importance.
        cov_test (np.ndarray): The covariates test data.
        m_test (np.ndarray): The m test data.
        n_sample (int): The number of samples.

    Returns:
        tuple[np.ndarray, np.ndarray]: The mean and standard deviation of the permutation importance.
    """
    rng = np.random.default_rng(random_seed)
    offestt_pred = sreft.model_1(np.concatenate((m_test, cov_test), axis=-1)).numpy()

    mean_pi = []
    std_pi = []
    n_pi = m_test.shape[1] + cov_test.shape[1]

    for i in range(n_pi):
        pis = []
        for j in range(n_sample):
            if i < m_test.shape[1]:
                m_test_rand = np.copy(m_test)
                rng.shuffle(m_test_rand[:, i])
                y_pred_rand = sreft.model_1(
                    np.concatenate((m_test_rand, cov_test), axis=-1)
                ).numpy()
            else:
                cov_test_rand = np.copy(cov_test)
                rng.shuffle(cov_test_rand[:, i - m_test.shape[1]])
                y_pred_rand = sreft.model_1(
                    np.concatenate((m_test, cov_test_rand), axis=-1)
                ).numpy()

            nglls_diff = (offestt_pred - y_pred_rand) ** 2
            temp_pi = np.nanmean(nglls_diff)
            pis.append(temp_pi)

        mean_pi.append(np.mean(pis))
        std_pi.append(np.std(pis))

    return np.array(mean_pi), np.array(std_pi)


def calculate_offsetT_prediction(
    sreft: tf.keras.Model,
    df: pd.DataFrame,
    scaled_features: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    scaler_y: sp.StandardScaler,
    name_biomarkers: list[str],
    getOffsetT: bool = True,
    getPrediction: bool = True,
) -> pd.DataFrame:
    """
    Calculate offsetT and prediction value of biomarkers.

    Args:
        sreft (tf.keras.Model): The trained SReFT model.
        df (pd.DataFrame): The input DataFrame.
        scaled_features (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): The scaled features. Pass x, cov, m, and y in that order.
        scaler_y (sp.StanderdScaler): The scaler for y.
        name_biomarkers (list[str]): List of biomarker names.

    Returns:
        pd.DataFrame: The DataFrame including the columns of the input DataFrame, offsetT and the prediction values.
    """
    df_ = df.copy()
    x_scaled, cov_scaled, m_scaled, y_scaled = scaled_features

    if getOffsetT:
        offsetT = sreft.model_1(np.concatenate((m_scaled, cov_scaled), axis=-1))
        df_ = df_.reset_index(drop=True).assign(offsetT=offsetT)

    if getPrediction:
        y_pred = pd.DataFrame(
            scaler_y.inverse_transform(sreft(scaled_features)),
            columns=[f"{biomarker}_pred" for biomarker in name_biomarkers],
        )
        df_ = df_.reset_index(drop=True).assign(**y_pred)

    return df_


class GompertzFitter(lifelines.fitters.ParametricUnivariateFitter):
    _fitted_parameter_names = ["lambda_", "c_"]

    def _cumulative_hazard(self, params, times):
        lambda_, c_ = params
        return lambda_ / c_ * (agnp.expm1(times * c_))


def survival_analysis(
    df: pd.DataFrame,
    surv_time: str,
    event: str,
    useOffsetT: bool = True,
    gompertz_init_params: list = [0.1, 0.1],
) -> dict:
    """
    Perform survival analysis and return a dictionary of survival analysis objects.


    If the survival time contains 0 or less, the survival time is converted so that the minimum value is 0.00001.

    Args:
        df (pd.DataFrame): Input DataFrame.
        surv_time (str): Column name of the survival time in df.
        event (str): Column name of the event in df.
        useOffsetT (bool, optional): Determines whether to use offsetT for the analysis. Defaults to True.

    Returns:
        dict: A dictionary of survival analysis objects.
    """
    fitters = [
        (lifelines.KaplanMeierFitter, "kmf", "KaplanMeier"),
        (lifelines.NelsonAalenFitter, "naf", "NelsonAalen"),
        (lifelines.ExponentialFitter, "epf", "Exponential"),
        (lifelines.WeibullFitter, "wbf", "Weibull"),
        (GompertzFitter, "gpf", "Gompertz"),
        (lifelines.LogLogisticFitter, "llf", "LogLogistic"),
        (lifelines.LogNormalFitter, "lnf", "LogNormal"),
    ]
    fit_model = {"title": event}
    if useOffsetT:
        df_surv = df[["ID", "offsetT", surv_time, event]].dropna().drop_duplicates()

        if df_surv["offsetT"].min() < 0:
            raise ValueError("offsetT must be greater than or equal to 0.")

        for fitter_class, key, label in fitters:
            if key == "gpf":
                fit_model[key] = fitter_class(label=label).fit(
                    durations=df_surv["offsetT"] + df_surv[surv_time],
                    event_observed=df_surv[event],
                    entry=df_surv["offsetT"],
                    initial_point=gompertz_init_params,
                )
            else:
                fit_model[key] = fitter_class(label=label).fit(
                    durations=df_surv["offsetT"] + df_surv[surv_time],
                    event_observed=df_surv[event],
                    entry=df_surv["offsetT"],
                )
    else:
        df_surv = df[["ID", surv_time, event]].dropna().drop_duplicates()
        for fitter_class, key, label in fitters:
            fit_model[key] = fitter_class(label=label).fit(
                durations=df_surv[surv_time], event_observed=df_surv[event]
            )

    return fit_model


def multi_column_filter(
    df: pd.DataFrame,
    upper_lim: dict[str, float] = None,
    lower_lim: dict[str, float] = None,
    IQR_filter: list = None,
):
    """
    Applies limits and IQR filtering on DataFrame columns.

    Operations:
        NaN substitution for values outside the specified upper and lower limits.
        IQR-based outlier removal in specified columns.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.
        upper_lim (dict[str, float], optional): Upper limits per column.
        lower_lim (dict[str, float], optional): Lower limits per column.
        IQR_filter (list, optional): Columns for IQR outlier detection

    Returns:
        pd.DataFrame: DataFrame after applying the defined filters.

    Notes:
        Overlapping `upper_lim`/`lower_lim` and `IQR_filter` keys cause warnings
        and filtering by `upper_lim`/`lower_lim`.
    """
    df_filtered = df.copy()
    if upper_lim is None:
        upper_lim = {}
    if lower_lim is None:
        lower_lim = {}
    if IQR_filter is None:
        IQR_filter = []

    if upper_lim:
        for k, v in upper_lim.items():
            df_filtered.loc[df_filtered[k] > v, k] = np.nan
        overlap_upper_IQR = set(upper_lim.keys()) & set(IQR_filter)
        if overlap_upper_IQR:
            warnings.warn(
                f"The columns {overlap_upper_IQR} were present in both upper_lim and IQR_filter, therefore they were filtered using the values from upper_lim."
            )

    if lower_lim:
        for k, v in lower_lim.items():
            df_filtered.loc[df_filtered[k] < v, k] = np.nan
        overlap_lower_IQR = set(lower_lim.keys()) & set(IQR_filter)
        if overlap_lower_IQR:
            warnings.warn(
                f"The columns {overlap_lower_IQR} were present in both lower_lim and IQR_filter, therefore they were filtered using the values from lower_lim."
            )

    if IQR_filter:
        IQR_exclusive = list(
            set(IQR_filter) - set(upper_lim.keys()) - set(lower_lim.keys())
        )
        q1 = df_filtered.quantile(0.25)
        q3 = df_filtered.quantile(0.75)
        iqr = q3 - q1
        df_filtered[IQR_exclusive] = df_filtered[IQR_exclusive].mask(
            (df_filtered < q1 - 1.5 * iqr) | (df_filtered > q3 + 1.5 * iqr), np.nan
        )

    return df_filtered


def calc_shap_explanation(
    sreft: tf.keras.Model,
    feature_names: list[str],
    cov_scaled: np.ndarray,
    m_scaled: np.ndarray,
) -> shap.Explanation:
    """
    Calculate the SHAP values for model 1.

    Args:
        sreft (tf.keras.Model): The model for which to calculate SHAP values.
        feature_names (list[str]): Provide the column names for 'm' and 'cov'. 'm' comes first, followed by 'cov'.
        cov_scaled (np.ndarray): The scaled covariate values.
        m_scaled (np.ndarray): The scaled m values.

    Returns:
        shap.Explanation: The explanation of SHAP values.
    """
    input1 = np.concatenate((m_scaled, cov_scaled), axis=-1)
    explainer_model_1 = shap.Explainer(
        sreft.model_1,
        input1,
        algorithm="permutation",
        seed=42,
        feature_names=feature_names,
    )
    shap_value_model_1 = explainer_model_1(input1)
    shap_exp_model_1 = shap.Explanation(
        shap_value_model_1.values,
        shap_value_model_1.base_values[0][0],
        shap_value_model_1.data,
        feature_names=feature_names,
    )

    return shap_exp_model_1


def load_shap(
    path_to_shap_file: str,
) -> shap.Explanation:
    """
    Load the specified SHAP binary file and return the SHAP explanations.

    Args:
        path_to_shap_file (str): The path to the SHAP file.

    Returns:
        Explanation: The explanation of SHAP values.
    """
    with open(path_to_shap_file, "rb") as p:
        shap_exp = pickle.load(p)

    return shap_exp


def save_shap(path_to_shap_file: str, shap_exp: shap.Explanation) -> None:
    """
    Save the SHAP explanations to the specified file.

    Parameters:
        path_to_shap_file (str): The path to save the SHAP file.
        shap_exp (shap.Explanation): The SHAP explanations to be saved.

    Returns:
        None
    """
    with open(path_to_shap_file, "wb") as p:
        pickle.dump(shap_exp, p)

    return None
