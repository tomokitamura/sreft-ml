import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn.preprocessing as sp
import tensorflow as tf
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde, linregress

from .make_demodata import model_sigmoid
from .utilities import clean_duplicate, n2mfrow


def hp_search_plot(
    df_grid: pd.DataFrame,
    eval_col: str = "score",
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Plot the results of hyperparameters search.

    Args:
        df_grid (pd.DataFrame): DataFrame containing the grid of hyperparameters.
        eval_col (str, optional): The column to use for evaluation. Defaults to "score".
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        plt.Figure: The plotted figure.
    """
    df_grid = df_grid.sort_values(eval_col, ascending=False).reset_index(drop=True)
    cols = df_grid.columns.tolist()
    cols.remove(eval_col)
    cols.append(eval_col)
    df_grid_sorted = df_grid[cols]

    string_columns = df_grid_sorted.select_dtypes(include="object").columns
    numeric_columns = df_grid_sorted.select_dtypes(exclude="object").columns

    for i in string_columns:
        df_grid_sorted[i] = sp.LabelEncoder().fit_transform(df_grid_sorted[i])

    scaler = sp.MinMaxScaler()
    df_grid_scaled = pd.DataFrame(
        scaler.fit_transform(df_grid_sorted), columns=df_grid_sorted.columns
    )

    cm = plt.get_cmap("seismic", 2)
    fig = plt.figure(tight_layout=True, dpi=300)
    for i, row in df_grid_scaled.iterrows():
        if i == len(df_grid_sorted) - 1:
            plt.plot(df_grid_scaled.columns, row.values, color=cm(1), lw=4)
        else:
            plt.plot(df_grid_scaled.columns, row.values, color=cm(0))

    for i in string_columns:
        label_unique = df_grid[i].unique()
        scaled_unique = df_grid_scaled[i].unique()
        for label_, scaled_ in zip(label_unique, scaled_unique):
            plt.text(
                i, scaled_, label_, ha="center", va="center", backgroundcolor="white"
            )

    for i in numeric_columns:
        min_val = df_grid_sorted[i].min()
        max_val = df_grid_sorted[i].max()
        plt.text(i, 0, min_val, ha="center", va="center", backgroundcolor="white")
        if min_val != max_val:
            plt.text(i, 1, max_val, ha="center", va="center", backgroundcolor="white")

    for i, val in enumerate(df_grid_scaled.iloc[-1, :]):
        col_name = df_grid_scaled.columns[i]
        if val not in [0, 1] and col_name in numeric_columns:
            plt.text(
                col_name,
                val,
                df_grid_sorted.iloc[-1, i],
                ha="center",
                va="center",
                backgroundcolor="white",
            )

    plt.xticks(rotation=45)
    plt.ylabel("Min-Max Normalized value")

    if save_file_path is not None:
        plt.savefig(save_file_path, transparent=True)

    return fig


def prediction_plot(
    sreft: tf.keras.Model,
    df: pd.DataFrame,
    name_biomarkers: list[str],
    name_covariates: list[str],
    scaler_y: sp.StandardScaler,
    scaler_cov: sp.StandardScaler,
    res: int = 100,
    density: bool = False,
    useOffsetT: bool = True,
    ncol_max: int = 4,
    colormap: ListedColormap = None,
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Plot the predictions of the SReFT model.

    Args:
        sreft (tf.keras.Model): The SReFT model.
        df (pd.DataFrame): DataFrame with the data.
        name_biomarkers (list[str]): The names of the biomarkers.
        name_covariates (list[str]): The names of the covariates.
        scaler_y (sp.StandardScaler): The scaler for the y values.
        scaler_cov (sp.StandardScaler): The scaler for the covariate values.
        res (int, optional): Resolution of the plot. Defaults to 100.
        density (bool, optional): Whether to plot density or not. Defaults to False.
        useOffsetT (bool, optional): Whether to use offsetT or not. Defaults to True.
        ncol_max (int, optional): Maximum number of columns for subplots. Defaults to 4.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        plt.Figure: The plotted figure.
    """
    n_biomarker = len(name_biomarkers)
    n_covariate = len(name_covariates)
    n_row, n_col = n2mfrow(n_biomarker, ncol_max)
    if colormap is None:
        colormap = plt.colormaps["Set1"]

    y_data = df[name_biomarkers].values
    if useOffsetT:
        x_data = df.TIME.values + df.offsetT.values
        cov_dummy = np.array([i for i in itertools.product([0, 1], repeat=n_covariate)])
        cov_dummy = np.repeat(cov_dummy, res, axis=0)
        cov_dummy_scaled = scaler_cov.transform(cov_dummy)
        x_model = np.linspace(x_data.min(), x_data.max(), res)
        x_model = np.tile(x_model, 2**n_covariate).reshape(-1, 1)
        x_model = np.concatenate((x_model, cov_dummy_scaled), axis=1)
        y_model = scaler_y.inverse_transform(sreft.model_y(x_model))
    else:
        x_data = df.TIME.values

    fig, axs = plt.subplots(
        n_row,
        n_col,
        figsize=(n_col * 3, n_row * 3),
        tight_layout=True,
        dpi=300,
        sharex="row",
    )
    for k, ax in enumerate(axs.flat):
        if k >= n_biomarker:
            ax.axis("off")
            continue

        if density:
            x_ = x_data[~np.isnan(y_data[:, k])]
            y_ = y_data[~np.isnan(y_data[:, k]), k]
            if np.var(x_) == 0:
                z = gaussian_kde(y_)(y_)
            else:
                xy = np.vstack([x_, y_])
                z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            ax.scatter(x_[idx], y_[idx], c=z[idx], s=2, label="_nolegend_")
        else:
            ax.scatter(x_data, y_data[:, k], c="silver", s=2, label="_nolegend_")

        if useOffsetT:
            for i in range(2**n_covariate):
                ax.plot(
                    x_model[res * i : res * (i + 1), 0],
                    y_model[res * i : res * (i + 1), k],
                    c=colormap(i),
                    lw=4,
                )
            ax.set_xlabel("Disease Time (year)")
        else:
            ax.set_xlabel("Observation Period (year)")

        ax.set_title(name_biomarkers[k], fontsize=15)

    if n_covariate > 0:
        legend_labels = [
            ", ".join(format(i, f"0{n_covariate}b")) for i in range(2**n_covariate)
        ]
        fig.legend(
            loc="center",
            framealpha=0,
            bbox_to_anchor=(1.1, 0.5),
            ncol=1,
            title=", ".join(name_covariates),
            labels=legend_labels,
        )

    if save_file_path is not None:
        fig.savefig(save_file_path, transparent=True, bbox_inches="tight")

    return fig


def get_regression_line_label(x: pd.Series, y: pd.Series) -> str:
    """
    Generate a label for a line fitted to the given x and y data using linear regression.

    Parameters:
        x (pd.Series): Series of x-axis data.
        y (pd.Series): Series of y-axis data.

    Returns:
        str: Label for the fitted line, including slope, intercept, and R-squared value.

    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    label_line_1 = rf"$y={slope:.3f}x{'' if intercept < 0 else '+'}{intercept:.3f}$"
    label_line_2 = rf"$R^2:{r_value**2:.2f}$"
    label_line = label_line_1 + "\n" + label_line_2

    return label_line


def single_panel_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: str | None = None,
    duplicate_key: list[str] | str | None = None,
    density: bool = False,
    identity: bool = False,
    save_file_path: str | None = None,
) -> sns.axisgrid.FacetGrid:
    """
    Draw a scatter plot using a single panel.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): X-axis column in df.
        y_col (str): Y-axis column in df.
        hue (str | None, optional): Column to stratify the plot. Defaults to None.
        duplicate_key (list[str] | str | None, optional): Specify the column name(s) from which duplicates are to be removed. Defaults to None.
        density (bool, optional): Whether to plot density. Defaults to False.
        identity (bool, optional): Whether to plot identity line. Defaults to False.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        sns.axisgrid.FacetGrid: FacetGrid object with the scatter plot.
    """
    if density:
        hue = None
        warnings.warn("Since density is True, the hue option is ignored.")

    if hue:
        df_ = clean_duplicate(df, [x_col, y_col, hue], duplicate_key)
        unique_hues = np.sort(df_[hue].unique())
        line_kws_ = None
    else:
        df_ = clean_duplicate(df, [x_col, y_col], duplicate_key)
        unique_hues = [None]
        line_kws_ = {"color": "red"}

    scatter_kws_ = {"alpha": 0.5, "s": 20, "edgecolor": "none"}
    if density:
        xy = df_[[x_col, y_col]].values.T
        z = gaussian_kde(xy)(xy)
        scatter_kws_.update({"c": z, "color": None, "cmap": "viridis"})

    g = sns.lmplot(
        data=df_,
        x=x_col,
        y=y_col,
        hue=hue,
        hue_order=unique_hues,
        scatter_kws=scatter_kws_,
        line_kws=line_kws_,
    )
    g.figure.set_dpi(300)

    if identity:
        if df[y_col].max() < df[x_col].min() or df[x_col].max() < df[y_col].min():
            warnings.warn(
                f"The data range of {x_col} and {y_col} is not covered, although idenntity=True. Skip drawing of identity line."
            )
        else:
            min_ = df[[x_col, y_col]].min().max()
            max_ = df[[x_col, y_col]].max().min()
            g.axes[0, 0].plot([min_, max_], [min_, max_], "k--")

    if hue:
        g.axes[0, 0].legend(
            ["_nolegend_", "dummy text", "_nolegned_"] * len(unique_hues)
        )
        for idx, h in enumerate(unique_hues):
            df_hue = df_.loc[df_[hue] == h]
            label_line = get_regression_line_label(df_hue[x_col], df_hue[y_col])
            g.axes[0, 0].get_legend().get_texts()[idx].set_text(label_line)
    else:
        label_line = get_regression_line_label(df_[x_col], df_[y_col])
        g.axes[0, 0].legend(labels=["_nolegend_", label_line])

    if save_file_path is not None:
        plt.savefig(save_file_path, transparent=True, dpi=300)

    return g


def multi_panel_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: list[str] | str,
    duplicate_key: list[str] | str | None = None,
    ncol_max: int = 4,
    density: bool = False,
    identity: bool = False,
    save_file_path: str | None = None,
) -> sns.axisgrid.FacetGrid:
    """
    Draw scatter plots with multiple panels based on stratification factors.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): X-axis column in df.
        y_col (str): Y-axis column in df.
        hue (list[str] | str): Columns to stratify the plot.
        duplicate_key (list[str] | str | None, optional): Specify the column name(s) from which duplicates are to be removed. Defaults to None.
        ncol_max (int, optional): Maximum number of columns. Defaults to 4.
        density (bool, optional): Whether to plot density. Defaults to False.
        identity (bool, optional): Whether to plot identity line. Defaults to False.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        sns.axisgrid.FacetGrid: FacetGrid object with the scatter plot.
    """
    if type(hue) is str:
        hue = [hue]

    df_ = clean_duplicate(df, [x_col, y_col] + hue, duplicate_key)
    hue_ = ", ".join(hue)
    if len(hue) > 1:
        df_[hue_] = df[hue].apply(lambda x: ", ".join(x.astype(str)), axis=1)
    unique_hue = np.sort(df_[hue_].unique())

    col_wrap = n2mfrow(len(df_[hue_].unique()), ncol_max=ncol_max)[1]
    g = sns.lmplot(
        data=df_,
        x=x_col,
        y=y_col,
        col=hue_,
        col_wrap=col_wrap,
        col_order=unique_hue,
        scatter=not density,
        scatter_kws={"alpha": 0.5, "s": 20, "edgecolor": "none"},
        line_kws={"color": "red", "label": "lines"},
    )
    g.figure.set_dpi(300)

    for idx, s in enumerate(unique_hue):
        df_hue = df_.loc[df_[hue_] == s]
        label_line = get_regression_line_label(df_hue[x_col], df_hue[y_col])

        if density:
            xy = df_hue[[x_col, y_col]].values.T
            z = gaussian_kde(xy)(xy)
            x_ = xy.T[:, :1]
            y_ = xy.T[:, 1:]
            g.axes[idx].scatter(x_, y_, c=z, s=20, edgecolor="none", cmap="viridis")
            g.axes[idx].legend([label_line])
        else:
            g.axes[idx].legend(["_nolegend_", label_line])

        if identity:
            if df[y_col].max() < df[x_col].min() or df[x_col].max() < df[y_col].min():
                warnings.warn(
                    f"The data range of {x_col} and {y_col} is not covered, although idenntity=True. Skip drawing of identity line."
                )
            else:
                min_ = df[[x_col, y_col]].min().max()
                max_ = df[[x_col, y_col]].max().min()
                g.axes[idx].plot([min_, max_], [min_, max_], "k--")

    if save_file_path is not None:
        plt.savefig(save_file_path, transparent=True, dpi=300)

    return g


def learning_history_plot(
    df_loss: pd.DataFrame, save_file_path: str | None = None
) -> plt.Figure:
    """
    Plot learning history.

    Args:
        df_loss (pd.DataFrame): Data frame converted from tf.keras.callbacks.History.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        plt.Figure: The plotted figure.
    """
    fig = plt.figure(tight_layout=True, dpi=300)
    plt.plot(df_loss["loss"], label="training")
    plt.plot(df_loss["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()

    if save_file_path is not None:
        plt.savefig(save_file_path, transparent=True)

    return fig


def histogram_plot(
    df: pd.DataFrame,
    col_name: list[str] | str,
    hue: str | None = None,
    sharex: bool = True,
    sharey: bool = True,
    ncol_max: int = 4,
    save_file_path: str | None = None,
) -> sns.axisgrid.FacetGrid:
    """
    Plot a stratified histogram by column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (list[str] | str): List of column names in df.
        hue (str | None, optional): Column to stratify the plot. Defaults to None.
        share{x, y} (bool | "col" | "row", optional): This is passed directly to seaborn.FacetGrid.
        ncol_max (int, optional): Maximum number of columns. Defaults to 4.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        sns.axisgrid.FacetGrid: FacetGrid object with the distribution plot.
    """
    if type(col_name) is str:
        col_name = [col_name]
    col_wrap = n2mfrow(len(col_name), ncol_max=ncol_max)[1]

    if hue is None:
        df_melt = pd.melt(df[col_name])
    else:
        df_melt = pd.melt(df[col_name + [hue]], hue)
    g = sns.FacetGrid(
        df_melt,
        col="variable",
        hue=hue,
        col_wrap=col_wrap,
        sharex=sharex,
        sharey=sharey,
        height=3.5,
    )
    g.map(plt.hist, "value", alpha=0.4)
    g.add_legend()
    g.set_titles("{col_name}")

    if save_file_path:
        g.savefig(save_file_path, transparent=True)

    return g


def scatter_matrix_plot(
    df: pd.DataFrame, save_file_path: str | None = None
) -> sns.axisgrid.PairGrid:
    """
    Plot correlation matrix.

    Args:
        df (pd.DataFrame): Input DataFrame.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        sns.axisgrid.PairGrid: PairGrid object with the correlation plot.
    """

    def corrfunc(x, y, **kwds):
        ax = plt.gca()
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
        r = x.corr(y, method="pearson")
        norm = plt.Normalize(-1, 1)
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

    g = sns.PairGrid(df)
    g.map_diag(sns.histplot, kde=False)
    g.map_lower(plt.scatter, s=2)
    g.map_upper(corrfunc)
    g.figure.tight_layout()

    if save_file_path:
        g.savefig(save_file_path)

    return g


def correlation_plot_strata(
    df: pd.DataFrame, name_biomarkers: list[str], strata: str = "status"
) -> None:
    """
    Generate a heatmap and pairplot of biomarkers for each strata.

    Args:
        df (pd.DataFrame): Dataframe with biomarkers and strata information.
        name_biomarkers (list[str]): List of biomarker names to plot.
        strata (str, optional): The name of the strata column. Default is 'status'.

    Returns:
        None
    """
    for i in range(2):
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            df[df[strata] == i][name_biomarkers].corr(),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            annot=True,
            fmt="1.2f",
        )

        plt.figure(figsize=(10, 10))
        sns.pairplot(df[df[strata] == i][name_biomarkers].reset_index(drop=True))

    sns.pairplot(df[name_biomarkers + [strata]], hue=strata, diag_kind="hist")
    return None


def residual_plot(
    df: pd.DataFrame,
    name_biomarkers: list[str],
    ncol_max: int = 4,
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Generate a plot of residuals.

    Args:
        df (pd.DataFrame): Input Dataframe. This must contain offsetT, actual value of biomarkers and prediction value of biomarkers.
        name_biomarkers (List[str]): List of biomarker names.
        ncol_max (int, optional): Maximum number of columns. Default is 4.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        Figure: The created matplotlib figure.
    """
    if not "offsetT" in df.columns:
        warnings.warn(
            "offsetT does not exist in df. df must contain offsetT. Skip drawing residual plot."
        )
        return None
    if not all([f"{biomarker}_pred" in df.columns for biomarker in name_biomarkers]):
        warnings.warn(
            "Some of the prediction values are missing in df. df must contain prediction values of biomarkers. Skip drawing residual plot."
        )
        return None
    n_biomarker = len(name_biomarkers)
    n_row, n_col = n2mfrow(n_biomarker, ncol_max)
    x_data = df.TIME.values + df.offsetT.values

    y_res = (
        df[[f"{biomarker}_pred" for biomarker in name_biomarkers]].values
        - df[name_biomarkers].values
    )

    fig, axs = plt.subplots(
        n_row, n_col, figsize=(n_col * 3, n_row * 3), tight_layout=True, dpi=300
    )
    for k, ax in enumerate(axs.flat):
        if k >= n_biomarker:
            ax.axis("off")
            continue

        ax.scatter(x_data, y_res[:, k], s=2)
        ax.axhline(0, c="black", ls="--")
        ax.set_title(name_biomarkers[k], fontsize=15)
        ax.set_xlabel("Disease Time (year)")
        ax.set_ylabel("y_pred - y_obs")

    if save_file_path is not None:
        fig.savefig(save_file_path, transparent=True)

    return fig


def permutation_importance_plot(
    mean_pi: np.ndarray,
    std_pi: np.ndarray,
    feature_label: list[str],
    y_axis_log: bool = False,
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Generate a permutation importance plot.

    Args:
        mean_pi (np.ndarray): Array of mean permutation importance values.
        std_pi (np.ndarray): Array of standard deviation permutation importance values.
        feature_label (list[str]): List of feature names for which PI was measured.
        y_axis_log (bool, optional): Whether to use log scale for y-axis. Default is False.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        plt.Figure: The plotted figure.
    """
    rank = np.argsort(mean_pi)
    fig = plt.figure(figsize=(len(rank) / 4, 10), dpi=300, tight_layout=True)
    plt.bar([feature_label[i] for i in rank], mean_pi[rank], yerr=std_pi[rank])
    plt.xticks(rotation=45, ha="right")
    if y_axis_log:
        plt.ylabel("Permutation Importance (log scale)")
        plt.yscale("log")
    else:
        plt.ylabel("Permutation Importance")

    if save_file_path is not None:
        plt.savefig(save_file_path, transparent=True)

    return fig


def merged_permutation_importance_plot(
    mean_pi: np.ndarray,
    name_biomarkers: list[str],
    name_covariates: list[str],
    y_axis_log: bool = False,
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Generate a permutation importance plot.

    Args:
        mean_pi (np.ndarray): Array of mean permutation importance values.
        name_biomarkers (List[str]): List of biomarker names.
        name_covariates (list[str]): The names of the covariates.
        y_axis_log (bool, optional): Whether to use log scale for y-axis. Default is False.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        plt.Figure: The plotted figure.
    """
    bar = pd.DataFrame(
        {
            "labels": [i + j for j in ["_slope", "_intercept"] for i in name_biomarkers]
            + name_covariates,
            "mean_pi": mean_pi.round(3),
        }
    )

    result_data = {"labels": [], "mean_pi": []}

    for i in name_biomarkers:
        slope_label = i + "_slope"
        intercept_label = i + "_intercept"
        slope_value = bar.loc[bar["labels"] == slope_label, "mean_pi"].values[0]
        intercept_value = bar.loc[bar["labels"] == intercept_label, "mean_pi"].values[0]
        total_value = slope_value + intercept_value
        result_data["labels"].append(i)
        result_data["mean_pi"].append(total_value)

    for i in name_covariates:
        cov_value = bar.loc[bar["labels"] == i, "mean_pi"].values[0]
        result_data["labels"].append(i)
        result_data["mean_pi"].append(cov_value)

    result_data = pd.DataFrame(result_data)

    mean_pi = result_data.mean_pi.values
    rank = np.argsort(mean_pi)
    fig = plt.figure(figsize=(len(rank) / 4, 10), dpi=300, tight_layout=True)
    plt.bar([result_data.labels.values[i] for i in rank], mean_pi[rank])
    plt.xticks(rotation=45, ha="right")
    if y_axis_log:
        plt.ylabel("Permutation Importance (log scale)")
        plt.yscale("log")
    else:
        plt.ylabel("Permutation Importance")

    if save_file_path is not None:
        plt.savefig(save_file_path, transparent=True)

    return fig


def var_y_plot(
    sreft: tf.keras.Model,
    name_biomarkers: list[str],
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Generate a plot of var_y.

    Args:
        sreft (tf.keras.Model): Object responsible for transforming the data.
        name_biomarkers (list[str]): List of biomarker names.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        plt.Figure: The plotted figure.
    """
    rank = np.argsort(np.exp(sreft.lnvar_y))
    fig = plt.figure(dpi=300, tight_layout=True)
    plt.barh([name_biomarkers[i] for i in rank], np.exp(sreft.lnvar_y)[rank])
    plt.gca().invert_yaxis()
    plt.xlabel("var_y")

    if save_file_path:
        plt.savefig(save_file_path, transparent=True)

    return fig


def prediction_sim_plot(
    df: pd.DataFrame,
    sreft: tf.keras.Model,
    params_true: pd.DataFrame,
    name_biomarkers: list[str],
    name_covariates: list[str],
    scaler_cov: sp.StandardScaler,
    scaler_y: sp.StandardScaler,
    res: int = 100,
    density: bool = False,
    ncol_max: int = 4,
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Generate a prediction simulation plot.

    Args:
        df (pd.DataFrame): Dataframe with biomarkers and other information.
        sreft (tf.keras.Model): Object responsible for transforming the data.
        params_true (pd.DataFrame): Dataframe with true parameters for the model.
        name_biomarkers (list[str]): List of biomarker names.
        name_covariates (list[str]): List of covariate names.
        scaler_cov: Scaler for the covariate values.
        scaler_y: Scaler for the y values.
        res (int, optional): Resolution for the plot. Default is 100.
        density (bool, optional): Whether to use density or not. Default is False.
        ncol_max (int, optional): Maximum number of columns for the plot. Default is 4.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        Figure: The created matplotlib figure.
    """
    n_biomarker = len(name_biomarkers)
    n_covariate = len(name_covariates)
    n_row, n_col = n2mfrow(n_biomarker, ncol_max)
    cm = plt.colormaps["Set1"]

    y_data = df[name_biomarkers].values
    x_data = df.TIME.values + df.offsetT.values

    cov_dummy = np.array(list(itertools.product([0, 1], repeat=n_covariate)))
    cov_dummy = np.repeat(cov_dummy, res, axis=0)
    cov_dummy_scaled = scaler_cov.transform(cov_dummy)
    x_model = np.linspace(x_data.min(), x_data.max(), res)
    input2 = np.tile(x_model, 2**n_covariate).reshape(-1, 1)
    input2 = np.concatenate((input2, cov_dummy_scaled), axis=1)
    y_model = scaler_y.inverse_transform(sreft.model_y(input2))

    name_covariates_true = [i for i in params_true.columns if "Covariate" in i]
    n_covariate_true = len(name_covariates_true)
    cov_dummy_true = np.array(list(itertools.product([0, 1], repeat=n_covariate_true)))
    cov_dummy_true = np.repeat(cov_dummy_true, res, axis=0)

    fig, axs = plt.subplots(
        n_row,
        n_col,
        figsize=(n_col * 3, n_row * 3),
        tight_layout=True,
        dpi=300,
        sharex="row",
    )
    for k, ax in enumerate(axs.flat):
        if k >= n_biomarker:
            ax.axis("off")
            continue

        if density:
            x_ = x_data[~np.isnan(y_data[:, k])]
            y_ = y_data[~np.isnan(y_data[:, k]), k]
            if np.var(x_) == 0:
                z = gaussian_kde(y_)(y_)
            else:
                xy = np.vstack([x_, y_])
                z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            ax.scatter(x_[idx], y_[idx], c=z[idx], s=2, label="_nolegend_")
        else:
            ax.scatter(x_data, y_data[:, k], c="silver", s=2, label="_nolegend_")

        pred_line = []
        for i in range(2**n_covariate):
            pred_line.extend(
                ax.plot(
                    x_model,
                    y_model[res * i : (res * i + res), k],
                    c=cm(i),
                    lw=3,
                )
            )

        true_line = []
        for i in range(2**n_covariate_true):
            y_true = model_sigmoid(
                x_model,
                cov_dummy_true[res * i : (res * i + res)],
                params_true.loc[k],
            )
            true_line.extend(
                ax.plot(
                    x_model,
                    y_true,
                    c=cm(i),
                    lw=3,
                    ls="dashed",
                )
            )

        ax.set_xlabel("Disease Time (year)")
        ax.set_title(name_biomarkers[k], fontsize=15)

    if n_covariate > 0:
        legend_labels = [
            ", ".join(format(i, f"0{n_covariate}b")) for i in range(2**n_covariate)
        ]
        fig.legend(
            handles=pred_line,
            loc="center",
            framealpha=0,
            bbox_to_anchor=(1.1, 0.7),
            title="Pred\n" + ", ".join(name_covariates),
            labels=legend_labels,
        )

        legend_labels_true = [
            ", ".join(format(i, f"0{n_covariate_true}b"))
            for i in range(2**n_covariate_true)
        ]
        fig.legend(
            handles=true_line,
            loc="center",
            framealpha=0,
            bbox_to_anchor=(1.1, 0.3),
            title="True\n" + ", ".join(name_covariates_true),
            labels=legend_labels_true,
        )

    if save_file_path:
        fig.savefig(save_file_path, transparent=True, bbox_inches="tight")

    return fig


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
        fig (plt.Figure): Matplotlib figure object representing the generated plot.
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


def scatter_matrix_plot_extra(
    df: pd.DataFrame, save_file_path: str | None = None
) -> sns.axisgrid.PairGrid:
    """
    Plot correlation matrix.

    Args:
        df (pd.DataFrame): Input DataFrame.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        sns.axisgrid.PairGrid: PairGrid object with the correlation plot.
    """

    def corrfunc(x, y, **kwds):
        ax = plt.gca()
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
        r = x.corr(y, method="pearson")
        norm = plt.Normalize(-1, 1)
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

    g = sns.PairGrid(df)
    g.map_diag(sns.histplot, kde=False)
    g.map_lower(plt.scatter, s=2)
    g.map_upper(corrfunc)
    g.figure.tight_layout()

    if save_file_path:
        g.savefig(save_file_path)

    return g


def shap_plots(
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
        plt.savefig(save_dir_path + "shap_bar.png", transparent=True)

    beeswarm_plot = plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)
    shap.plots.beeswarm(shap_exp_model_1, show=False)
    if save_dir_path:
        plt.savefig(save_dir_path + "shap_beeswarm.png", transparent=True)

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
        fig.savefig(save_dir_path + "shap_dependence.png", transparent=True)

    return bar_plot, beeswarm_plot, fig


def merged_shap_bar_plot(
    shap_exp_model_1: shap.Explanation,
    name_biomarkers: list[str],
    name_covariates: list[str],
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Plot the SHAP values of the model 1.

    Args:
        shap_exp_model_1 (shap.Explanation): The SHAP explanation for the model 1.
        name_biomarkers (List[str]): List of biomarker names.
        name_covariates (list[str]): The names of the covariates.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        fig (plt.Figure): Matplotlib figure object representing the generated plot.
    """
    bar = pd.DataFrame(
        {
            "labels": [i + j for j in ["_slope", "_intercept"] for i in name_biomarkers]
            + name_covariates,
            "shap": np.mean(abs(shap_exp_model_1.values), axis=0).round(3),
        }
    )

    result_data = {"labels": [], "shap": []}

    for i in name_biomarkers:
        slope_label = i + "_slope"
        intercept_label = i + "_intercept"
        slope_value = bar.loc[bar["labels"] == slope_label, "shap"].values[0]
        intercept_value = bar.loc[bar["labels"] == intercept_label, "shap"].values[0]
        total_value = slope_value + intercept_value
        result_data["labels"].append(i)
        result_data["shap"].append(total_value)

    for i in name_covariates:
        cov_value = bar.loc[bar["labels"] == i, "shap"].values[0]
        result_data["labels"].append(i)
        result_data["shap"].append(cov_value)

    result_data = pd.DataFrame(result_data)

    shap_exp = result_data.shap.values
    rank = np.argsort(shap_exp)
    fig = plt.figure(figsize=(len(rank) / 4, 10), dpi=300, tight_layout=True)
    plt.bar([result_data.labels.values[i] for i in rank], shap_exp[rank])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean(|SHAP value|)")

    if save_file_path is not None:
        plt.savefig(save_file_path, transparent=True)

    return fig
