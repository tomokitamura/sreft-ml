import math
import subprocess
import warnings

import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import statsmodels.formula.api as smf
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import importlib.util
import types
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import scipy.stats as stats
from lifelines.utils import concordance_index



class NullModel:
    def __init__(self, Intercept, TIME):
        self.params = [Intercept, TIME]


def n2mfrow(n_plots: int, ncol_max: int = 3) -> tuple[int, int]:
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

    return x, cov, m, y, linreg


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


def calculate_offsetT_prediction(
    sreft: tf.keras.Model,
    df: pd.DataFrame,
    scaled_features: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    scaler_y: sp.StandardScaler,
    name_biomarkers: list[str],
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
    offsetT = sreft.model_1(np.concatenate((m_scaled, cov_scaled), axis=-1))
    y_pred = pd.DataFrame(
        scaler_y.inverse_transform(sreft(scaled_features)),
        columns=[f"{biomarker}_pred" for biomarker in name_biomarkers],
    )
    df_ = df_.assign(offsetT=offsetT, **y_pred)
    return df_


def remove_outliers(data):
    """
    Removes outliers from data based on interquartile range.
    """
    valid_data = data[~np.isnan(data)]  # Exclude NaN values for the calculation
    if len(valid_data) == 0:  # Check for empty data
        return np.array([], dtype=bool)

    Q1 = np.percentile(valid_data, 25)
    Q3 = np.percentile(valid_data, 75)
    IQR = Q3 - Q1
    mask = (data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)
    return mask




def import_if_installed(module_name: str) -> types.ModuleType | None:
    """
    Imports a module if it is installed.

    Args:
        module_name (str): The name of the module to import.

    Returns:
        module: The imported module. If the module is not installed, None is returned.
    """
    if importlib.util.find_spec(module_name) is not None:
        return importlib.import_module(module_name)
    else:
        warnings.warn(f"Module {module_name} is not installed.")
        return None


shap = import_if_installed("shap")
if shap is not None:
    import pickle

    def calc_shap_value_model_1(
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

    def calc_shap_value_model_y(
        sreft: tf.keras.Model,
        name_covariates: list[str],
        x_scaled: np.ndarray,
        cov_scaled: np.ndarray,
        m_scaled: np.ndarray,
    ) -> shap.Explanation:
        """
        Calculate the SHAP values for model y.

        Args:
            sreft (tf.keras.Model): The model for which to calculate SHAP values.
            name_biomarkers (list[str]): List of baiomarkers' names.
            name_covariates (list[str]): List of covariates' names.
            x_scaled (np.ndarray): The scaled x values.
            cov_scaled (np.ndarray): The scaled covariate values.
            m_scaled (np.ndarray): The scaled m values.

        Returns:
            shap.Explanation: The explanation of SHAP values.
        """
        offsetT = sreft.model_1(np.concatenate((m_scaled, cov_scaled), axis=-1)).numpy()
        dis_time = offsetT + x_scaled
        input2 = np.concatenate((dis_time, cov_scaled), axis=-1)

        explainer_model_y = shap.Explainer(
            sreft.model_y,
            input2,
            algorithm="permutation",
            seed=42,
            feature_names=["TIME"] + name_covariates,
        )
        shap_value_model_y = explainer_model_y(input2)
        shap_exp_model_y = shap.Explanation(
            shap_value_model_y.values,
            shap_value_model_y.base_values[0][0],
            shap_value_model_y.data,
            feature_names=["TIME"] + name_covariates,
        )

        return shap_exp_model_y

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

    def shap_model_1_plot(
        shap_exp_model_1: shap.Explanation,
        ncol_max: int = 4,
        save_dir_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
        """
        Plot the SHAP values of the model 1.

        Args:
            shap_exp_model_1 (shap.Explanation): The SHAP explanation for the model 1.
            ncol_max (int, optional): Maximum number of columns for subplots. Defaults to 4.
            save_dir_path (str, optional): The path where the plot will be saved. Default to None.

        Returns:
            tuple[plt.Figure, plt.Figure, plt.Figure]: Plot objects for shap bar, beeswarm and dependence plot.
        """

        bar_plot = plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)
        shap.plots.bar(shap_exp_model_1, show=False)
        plt.title("model 1")
        if save_dir_path:
            plt.savefig(save_dir_path + "shap_bar_model_1.png", transparent=True)

        beeswarm_plot = plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)
        shap.plots.beeswarm(shap_exp_model_1, show=False)
        if save_dir_path:
            plt.savefig(save_dir_path + "shap_beeswarm_model_1.png", transparent=True)

        # plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)
        # shap.plots.waterfall(shap_exp_model_1[0], show=not save_fig)
        # if save_dir_path:
        #     plt.savefig(save_dir_path + "shap_waterfall_model_1.png", transparent=True)
        #     plt.show()

        n_row, n_col = n2mfrow(shap_exp_model_1.shape[1], ncol_max=ncol_max)
        fig, axs = plt.subplots(
            n_row,
            n_col,
            figsize=(n_col * 4, n_row * 3),
            tight_layout=True,
            dpi=300,
        )
        for k, ax in enumerate(axs.flat):
            if k >= shap_exp_model_1.shape[1]:
                ax.axis("off")
                continue
            shap.plots.scatter(
                shap_exp_model_1[:, k],
                color=shap_exp_model_1,
                x_jitter=0.01,
                ax=ax,
                show=False,
            )
        fig.suptitle("model 1")
        if save_dir_path:
            fig.savefig(save_dir_path + "shap_dependence_model_1.png", transparent=True)

        return bar_plot, beeswarm_plot, fig

    def shap_model_y_plot(
        shap_exp_model_y: shap.Explanation,
        name_biomarkers: list[str],
        name_covariates: list[str],
        ncol_max: int = 4,
        save_dir_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Figure]:
        """
        Plot SHAP model Y.

        Args:
            shap_exp_model_y (shap.Explanation): SHAP explanation model for Y.
            name_biomarkers (list[str]): List of biomarker names.
            name_covariates (list[str]): List of covariate names.
            ncol_max (int, optional): Maximum number of columns for subplots. Defaults to 4.
            save_dir_path (str, optional): The path where the plot will be saved. Default to None.

        Returns:
            tuple[plt.Figure, plt.Figure, plt.Figure]: Plot objects for shap bar plot.
        """
        n_biomarker = len(name_biomarkers)
        # n_covariate = len(name_covariates)
        n_row, n_col = n2mfrow(n_biomarker, ncol_max=ncol_max)

        shap_data = np.mean(abs(shap_exp_model_y.values), axis=0)
        bar_plot_fig, bar_plot_axs = plt.subplots(
            n_row,
            n_col,
            figsize=(n_col * 3, n_row * 3),
            sharex=True,
            tight_layout=True,
            dpi=300,
        )
        for k, ax in enumerate(bar_plot_axs.flat):
            if k >= n_biomarker:
                ax.axis("off")
                continue

            ax.barh(["TIME"] + name_covariates, shap_data[:, k])
            ax.set_title(name_biomarkers[k])
            ax.set_xlabel("mean(|SHAP value|)")
            ax.invert_yaxis()
        bar_plot_fig.suptitle("model y")
        if save_dir_path:
            plt.savefig(save_dir_path + "shap_bar_model_y.png", transparent=True)

        shap_data_overall = np.mean(abs(shap_exp_model_y.values), axis=(0, 2))
        bar_all_plot = plt.figure(dpi=300, tight_layout=True)
        plt.barh(["TIME"] + name_covariates, shap_data_overall)
        plt.xlabel("mean(|SHAP value|)")
        plt.gca().invert_yaxis()
        plt.suptitle("model y")
        if save_dir_path:
            plt.savefig(
                save_dir_path + "shap_bar_model_y_overall.png", transparent=True
            )

        # for i in range(n_biomarker):
        #     tmp_beeswarm_plot = plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)
        #     shap.plots.beeswarm(shap_exp_model_y[:, :, i], show=False)
        #     plt.title(name_biomarkers[i])
        #     if save_dir_path:
        #         plt.savefig(
        #             save_dir_path + "shap_beeswarm_model_y_" + name_biomarkers[i] + ".png",
        #             transparent=True,
        #         )

        # for i in range(n_biomarker):
        #     plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)
        #     shap.plots.waterfall(shap_exp_model_y[0, :, i], show=not save_fig)
        #     plt.title(name_biomarkers[i])
        #     if save_dir_path:
        #         plt.savefig(
        #             save_dir_path + "shap_waterfall_model_y_" + name_biomarkers[i] + ".png",
        #             transparent=True,
        #         )

        # for i in range(n_covariate + 1):
        #     fig, axs = plt.subplots(
        #         n_row,
        #         n_col,
        #         figsize=(n_col * 4, n_row * 3),
        #         tight_layout=True,
        #         dpi=300,
        #     )
        #     for k, ax in enumerate(axs.flat):
        #         if k >= n_biomarker:
        #             ax.axis("off")
        #             continue
        #         shap.plots.scatter(
        #             shap_exp_model_y[:, i, k],
        #             color=shap_exp_model_y[:, :, k],
        #             x_jitter=0.01,
        #             title=name_biomarkers[k],
        #             ax=ax,
        #             show=False,
        #         )
        #     fig.suptitle("model y")
        #     if save_dir_path:
        #         name_input2 = ["TIME"] + name_covariates
        #         fig.savefig(
        #             save_dir_path + "shap_dependence_model_y_" + name_input2[i] + ".png",
        #             transparent=True,
        #         )

        return bar_plot_fig, bar_all_plot


lifelines = import_if_installed("lifelines")
if lifelines is not None:
    import autograd.numpy as agnp

    class GompertzFitter(lifelines.fitters.ParametricUnivariateFitter):
        _fitted_parameter_names = ["lambda_", "c_"]

        def _cumulative_hazard(self, params, times):
            lambda_, c_ = params
            return lambda_ / c_ * (agnp.expm1(times * c_))

    def survival_analysis(
        df: pd.DataFrame, surv_time: str, event: str, useOffsetT: bool = True
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
                        initial_point=[0.1, 0.1],
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

    def surv_analysis_plot(
        fit_model: dict,
        ci_show: bool = True,
        only_best: bool = True,
        title: str | None = None,
        xlabel: str = "Disease Time (year)",
        save_dir_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
        """
        Generate survival analysis plot.

        Args:
            fit_model (dict): A dictionary of survival analysis objects.
            ci_show (bool, optional): Whether to show confidence intervals. Defaults to True.
            only_best (bool, optional): Whether to plot only the best model. Defaults to True.
            title (str | None, optional): Title for each plot. If None, the title from fit_model is used.
            xlabel (str, optional): X-axis label for each plot. Defaults to "Disease Time (year)".
            save_dir_path (str, optional): The path where the plot will be saved. Default to None.

        Returns:
            tuple[plt.Figure, plt.Figure, plt.Figure]: Plot objects for the survival function, cumulative hazard function, and hazard function.
        """
        if title is None:
            title = fit_model["title"]

        fit_model_parametric = {
            key: value
            for key, value in fit_model.items()
            if key not in ["title", "kmf", "naf"]
        }
        if only_best:
            aics = [i.AIC_ for i in fit_model_parametric.values()]
            best_model = list(fit_model_parametric.keys())[aics.index(min(aics))]
            fit_model_parametric = {best_model: fit_model_parametric[best_model]}

        surv_plot = plt.figure(figsize=(5, 5), dpi=300)
        fit_model["kmf"].plot_survival_function(ci_show=ci_show, lw=2)
        [
            k.plot_survival_function(ci_show=ci_show, lw=2)
            for k in fit_model_parametric.values()
        ]
        plt.xlabel(xlabel)
        plt.ylabel("Survival Function")
        plt.title(title)
        if save_dir_path:
            plt.savefig(save_dir_path + "surv_func.png", transparent=True)

        cumhaz_plot = plt.figure(figsize=(5, 5), dpi=300)
        fit_model["naf"].plot_cumulative_hazard(ci_show=ci_show, lw=2)
        [
            k.plot_cumulative_hazard(ci_show=ci_show, lw=2)
            for k in fit_model_parametric.values()
        ]
        plt.xlabel(xlabel)
        plt.ylabel("Cumlative Hazard Function")
        plt.title(title)
        if save_dir_path:
            plt.savefig(save_dir_path + "cumhaz_func.png", transparent=True)

        haz_plot = plt.figure(figsize=(5, 5), dpi=300)
        fit_model["naf"].plot_hazard(bandwidth=2, ci_show=ci_show, lw=2)
        [k.plot_hazard(ci_show=ci_show, lw=2) for k in fit_model_parametric.values()]
        plt.xlabel(xlabel)
        plt.ylabel("Hazard Function")
        plt.title(title)
        if save_dir_path:
            plt.savefig(save_dir_path + "haz_func.png", transparent=True)

        return surv_plot, cumhaz_plot, haz_plot
    
    def calculate_c_index(df, time_col, event_col, predicted_col, id_col):
        """
        Calculate the concordance index (c-index) for each subject in the dataframe.
    
        Parameters:
            df (pd.DataFrame): Input dataframe containing time, event, predicted scores, and subject ID.
            time_col (str): Column name for the observed survival times or event times.
            event_col (str): Column name for the event indicator (1 if event occurred, 0 if censored).
            predicted_col (str): Column name for the predicted risk scores.
            id_col (str): Column name for the subject ID.
    
        Returns:
            float: The overall concordance index (c-index).
        """
        unique_ids = df[id_col].unique()
        all_event_times = []
        all_predicted_scores = []
        all_events = []
    
        for uid in unique_ids:
            subject_data = df[df[id_col] == uid]
            event_times = subject_data[time_col].values
            predicted_scores = subject_data[predicted_col].values
            events = subject_data[event_col].values
    
            all_event_times.extend(event_times)
            all_predicted_scores.extend(predicted_scores)
            all_events.extend(events)
    
        return concordance_index(all_event_times, all_predicted_scores, all_events)
    



minepy = import_if_installed("minepy")


def scatter_matrix_plot_extra(
    df: pd.DataFrame, upper_type: str, save_file_path: str | None = None
) -> sns.axisgrid.PairGrid:
    """
    Plot correlation matrix.

    Args:
        df (pd.DataFrame): Input DataFrame.
        upper_type (str): Type of correlation calculation. ("corr" or "mic")
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        sns.axisgrid.PairGrid: PairGrid object with the correlation plot.
    """

    def corrfunc(x, y, upper_type, **kwds):
        ax = plt.gca()
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
        if upper_type == "corr":
            r = x.corr(y, method="pearson")
            norm = plt.Normalize(-1, 1)
        elif upper_type == "mic":
            mine = minepy.MINE()
            mine.compute_score(x, y)
            r = mine.mic()
            norm = plt.Normalize(0, 1)
        facecolor = plt.get_cmap("seismic")(norm(r))
        ax.set_facecolor(facecolor)
        ax.set_alpha(0)
        lightness = (max(facecolor[:3]) + min(facecolor[:3])) / 2
        ax.annotate(
            f"{r:.2f}",
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            color="white" if lightness < 0.7 else "black",
            size=26,
            ha="center",
            va="center",
        )

    if upper_type not in ["corr", "mic"]:
        raise ValueError(
            "The upper_type you specified is not appropriate. Please specify 'corr' or 'mic'."
        )
    if upper_type == "mic" and minepy is None:
        raise ValueError("'mic' is selected, but minepy is not installed.")
    g = sns.PairGrid(df)
    g.map_diag(sns.histplot, kde=False)
    g.map_lower(plt.scatter, s=2)
    g.map_upper(corrfunc, upper_type=upper_type)
    g.figure.suptitle(upper_type, fontsize=26)
    g.figure.tight_layout()

    if save_file_path:
        g.savefig(save_file_path)

    return g


def r_squared_plot(
    df: pd.DataFrame,
    name_biomarkers: list[str],
    isSort: bool = True,
    cutoff: float = 0.1,
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Generate a horizontal bar plot displaying the R-squared values of biomarkers.

    Args:
        df (pd.DataFrame): DataFrame containing the biomarker data.
        name_biomarkers (list[str]): List of column names representing the biomarkers.
        isSort (bool, optional): If True, sort biomarkers by R-squared values. Default is True.
        cutoff (float, optional): Cutoff value for highlighting specific R-squared values. Biomarkers with R-squared values greater than or equal to cutoff will be highlighted. Default is 0.1.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
    - fig (plt.Figure): Matplotlib figure object representing the generated plot.
    """
    res = df[name_biomarkers].values - df.filter(like="_pred", axis=1).values
    res_var = np.nanvar(res, axis=0)
    df_var = np.nanvar(df[name_biomarkers].values, axis=0)
    r_squared = 1 - res_var / df_var

    cm = plt.get_cmap("tab10")
    fig = plt.figure(dpi=300, tight_layout=True)
    if cutoff > 0:
        plt.axvline(x=cutoff, ls="--", c="black")
        colors = [cm(1) if x >= cutoff else cm(0) for x in r_squared]
    else:
        colors = [cm(0) for _ in range(len(r_squared))]

    if isSort:
        rank = np.argsort(r_squared)
        plt.barh(
            [name_biomarkers[i] for i in rank],
            r_squared[rank],
            color=[colors[i] for i in rank],
        )
    else:
        plt.barh(name_biomarkers, r_squared, color=colors)

    plt.xlabel("r_squared")

    if save_file_path:
        plt.savefig(save_file_path, transparent=True)

    return fig

def All_raw_data_plots(df: pd.DataFrame,
                   name_biomarkers: list,
                   group_column_name: str | None = "ARM",
                   placebo_group_name: str | None = "プラセボ群",
                   dose_group_name: str | None = "治療群",
                   placebo_group_number: int | None = 1,
                   dose_group_number: int | None = 0,
                   density: bool=True,
                   title: str | None = None,
                   save_fig: bool=True, 
                   save_path: str | None = None
                   ):


    ave_df_0 = df[df[group_column_name] == dose_group_number].groupby("TIME", as_index=False).mean(numeric_only=True)
    ave_df_1 = df[df[group_column_name] == placebo_group_number].groupby("TIME", as_index=False).mean(numeric_only=True)

    n_row, n_col = n2mfrow(len(name_biomarkers), ncol_max=5)
    fig, axs = plt.subplots(n_row, n_col, figsize = (n_col*5, n_row*5+1))
    for i, ax in enumerate(axs.flatten()):
        if i>=len(name_biomarkers):
            ax.axis("off")
            continue
        ax.plot(ave_df_0.TIME, ave_df_0[name_biomarkers[i]], color="#ff0000", lw=3, label="強化療法群")
        ax.plot(ave_df_1.TIME, ave_df_1[name_biomarkers[i]], color="#000080", lw=3, label="プラセボ群")
        
        df_ = df[["TIME", name_biomarkers[i]]].dropna(subset=["TIME", name_biomarkers[i]])
        
        x_data_tmp = df_.TIME.values
        y_data_tmp = df_[name_biomarkers[i]].values

        if density:
            x_ = x_data_tmp
            y_ = y_data_tmp
            if np.var(x_) == 0:
                z = gaussian_kde(y_)(y_)
            else:
                xy = np.vstack([x_, y_])
                z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            ax.scatter(x_[idx], y_[idx], c=z[idx], s=2, label="_nolegend_")
        else:
            ax.scatter(
                x_data_tmp, y_data_tmp[:, i], c="silver", s=2, label="_nolegend_"
            )
        ax.fill_between(ave_df_0.TIME, ave_df_0[name_biomarkers[i]] - np.std(ave_df_0[name_biomarkers[i]]), ave_df_0[name_biomarkers[i]] + np.std(ave_df_0[name_biomarkers[i]]), alpha=0.2, color="red")
        ax.fill_between(ave_df_1.TIME, ave_df_1[name_biomarkers[i]] - np.std(ave_df_1[name_biomarkers[i]]), ave_df_1[name_biomarkers[i]] + np.std(ave_df_1[name_biomarkers[i]]), alpha=0.2, color="blue")
        ax.set_title(name_biomarkers[i], fontsize=20, fontweight=1000)
        dif = max(ave_df_0[name_biomarkers[i]].max(), ave_df_1[name_biomarkers[i]].max()) - min(ave_df_0[name_biomarkers[i]].min(), ave_df_1[name_biomarkers[i]].min())
        ax.set_ylim(min(ave_df_0[name_biomarkers[i]].min(), ave_df_1[name_biomarkers[i]].min()) - dif, max(ave_df_0[name_biomarkers[i]].max(), ave_df_1[name_biomarkers[i]].max()) + dif)
        ax.legend(loc="best", fontsize=10)
        plt.rcParams["font.size"] = 20
    plt.suptitle(title, fontsize=30, fontweight=1000)
    plt.tight_layout()
    plt.show()
    if save_fig:
        fig.savefig(save_path + "All_care_plots.png", transparent=True, bbox_inches="tight")
    
    return fig


def each_data_count(df: pd.DataFrame,
                   name_biomarkers: list,
                   plot: bool = True,
                   save_path: str | None = None,
                   group_column_name: str | None = "ARM",
                   placebo_group_number: int | None = 1,
                   dose_group_number: int | None = 0
                   ):
    
    def add_value_label(x_list, y_list):
        for i in range(1, len(x_list) + 1):
            ax.text(i - 1, 0.7*y_list[i - 1], y_list[i - 1], ha="center", fontsize=8, fontweight="bold", bbox={"facecolor" : "white","boxstyle" : "Round",})
    
    for p in range(2):
        
        tmp_df = pd.DataFrame(df[df[group_column_name] == df.groupby(["TIME"], as_index=False).count()])
        tmp_df_index_lis = list(map(str, [round(list(tmp_df["TIME"])[n], 2) for n in range(len(tmp_df["TIME"]))]))
        tmp_df = tmp_df[name_biomarkers]
    
        n_row, n_col = n2mfrow(len(name_biomarkers), ncol_max=5)
        fig, axs = plt.subplots(n_row, n_col, figsize = (20, 15))

    for i, ax in enumerate(axs.flatten()):
        if i>=len(name_biomarkers):
            ax.axis("off")
            continue
        ax.bar(tmp_df_index_lis, tmp_df[name_biomarkers[i]], width=0.4)
        ax.set_xlabel("Time Point (year)", fontdict={"fontsize":15, "fontweight":"bold"})
        ax.set_title(name_biomarkers[i], fontsize=20)
        ax.tick_params(labelsize=8)
        ax.set_ylim(0, 900)
        if plot:
            ax.plot(tmp_df_index_lis, tmp_df[name_biomarkers[i]])
        add_value_label(tmp_df_index_lis, list(tmp_df[name_biomarkers[i]]))
    fig.suptitle("各タイムポイントにおける患者数", fontsize=25)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    if save_path!= None:
        fig.savefig(save_path + "each_data_count.png", transparent=True, bbox_inches="tight", dpi=600)
    
    return fig



def between_group_comparison_plot(
    df, 
    title='NYHA分類/offsetTをNYHAと同じ例数ずつ分けた時の分布', 
    xlabel='NYHA分類', 
    ylabel='死亡率', 
    legend_x='NYHA分類', 
    legend_y='offsetT', 
    save_fig=False, 
    save_path='NYHA_offsetT_bar_plot.png'
):
    """
    Generate a moltarity comparison of NYHA and offsetT.

    Args:
        df (pd.DataFrame): DataFrame containing the biomarker data.
        title The title of output graph.
        xlabel The name of x-label.
        ylabel The name of y-label.
        legend_x The displayed legend for x-label.
        legend_y The displayed legend for y-label.
        save_fig A bool value which to save or not.
        save_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
    - fig (plt.Figure): Matplotlib figure object representing the generated plot.
    """
    # NYHA2とNYHA3の患者をそれぞれ2つのグループに分割
    def split_nyha_groups(df, nyha_value, group_labels):
        subset = df[df['NYHA分類'] == nyha_value]
        subset = subset.sort_values(by='NTproBNP_log')
        n = len(subset) // 2
        remainder = len(subset) % 2
        groups = []
        for i in range(2):
            size = n + (1 if i < remainder else 0)
            groups.append(subset.iloc[:size])
            subset = subset.iloc[size:]
        for group, label in zip(groups, group_labels):
            df.loc[group.index, 'NYHA分類'] = label

    split_nyha_groups(df, 2, ['2', '2+'])
    split_nyha_groups(df, 3, ['3', '3+'])

    # データのグループ化
    df_baseline = df.drop_duplicates(subset="ID")
    groups = ['1.0', '2', '2+', '3', '3+', '4.0']

    # 各グループのデータを確認
    for group in groups:
        print(f"Group {group}:")
        print(df_baseline[df_baseline['NYHA分類'] == group].head())

    # NYHA分類カラムを文字列型に変換
    df_baseline['NYHA分類'] = df_baseline['NYHA分類'].astype(str)

    # NYHA分類ごとの人数と死亡者数を計算
    nyha_counts = [len(df_baseline[df_baseline['NYHA分類'] == group]) for group in groups]
    nyha_deaths = [
        len(df_baseline[(df_baseline['NYHA分類'] == group) & (df_baseline['DEATH'] == 1)]) for group in groups
    ]

    # 各グループの人数と死亡者数を確認
    print("NYHA counts:", nyha_counts)
    print("NYHA deaths:", nyha_deaths)

    # offsetTを用いた人数と死亡者数の計算
    offsetT_df = df_baseline.sort_values('offsetT')
    offsetT_counts = []
    offsetT_deaths = []
    offsetT_ranges = []
    cumulative_count = 0

    for count in nyha_counts:
        if count > 0:
            group = offsetT_df.iloc[cumulative_count:cumulative_count + count]
            offsetT_counts.append(len(group))
            offsetT_deaths.append(len(group[group['DEATH'] == 1]))
            offsetT_ranges.append(f"{group['offsetT'].min():.2f}-{group['offsetT'].max():.2f}")
            cumulative_count += count
        else:
            offsetT_counts.append(0)
            offsetT_deaths.append(0)
            offsetT_ranges.append("N/A")

    # 各グループのoffsetTによる人数と死亡者数を確認
    print("OffsetT counts:", offsetT_counts)
    print("OffsetT deaths:", offsetT_deaths)
    print("OffsetT ranges:", offsetT_ranges)

    # 死亡率の計算
    nyha_death_rates = [death / count if count > 0 else 0 for death, count in zip(nyha_deaths, nyha_counts)]
    offsetT_death_rates = [death / count if count > 0 else 0 for death, count in zip(offsetT_deaths, offsetT_counts)]

    # 死亡率の確認
    print("NYHA death rates:", nyha_death_rates)
    print("OffsetT death rates:", offsetT_death_rates)

    # カイ二乗検定の実行
    chi2, p, dof, expected = stats.chi2_contingency([
        nyha_deaths, 
        offsetT_deaths
    ])

    print(f"カイ二乗値: {chi2}, p値: {p}, 自由度: {dof}")

    # グラフの描画
    fig, ax = plt.subplots(figsize=(10, 8))
    x = range(len(groups))
    width = 0.35

    ax.bar([p - width/2 for p in x], nyha_death_rates, width, label=legend_x)
    ax.bar([p + width/2 for p in x], offsetT_death_rates, width, label=legend_y)

    ax.set_xlabel(xlabel, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=20)
    ax.set_xticks(x, fontsize=15)
    ax.set_xticklabels(groups)
    ax.legend(fontsize=20)

    # グラフの下に人数とoffsetTの範囲を表示
    for i in range(len(groups)):
        ax.text(i, -0.15, f'{nyha_counts[i]} 例', ha='center', va='bottom', fontsize=10, fontweight="bold")
        ax.text(i, -0.05, f'{offsetT_ranges[i]} 年', ha='center', va='top', fontsize=10, rotation=30)
    ax.text(-0.1, -0.15, "疾患時間範囲",
            transform=ax.transAxes,
            fontsize=12)
    
    ax.text(-0.1, -0.28, "例数",
            transform=ax.transAxes,
            fontsize=12)
    
    ax.text(0.95, 0.98, f'カイ二乗値: {chi2:.3f}, p値: {p:.3f}, 自由度: {dof}', 
            transform=ax.transAxes, 
            verticalalignment='top', 
            horizontalalignment='right',
            fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_path)

    plt.show()

