# %% Libraries, Functions
from collections.abc import Callable

import numpy as np
import pandas as pd


def generate_params_cov(
    rng: np.random.Generator, n_biomarker_total: int, n_covariate_total: int
) -> np.ndarray:
    """
    Generate a matrix of covariate parameters for a statistical model.

    Args:
        rng (np.random.Generator): A random number generator.
        n_biomarker_total (int): The number of biomarkers to consider.
        n_covariate_total (int): The number of covariates to generate parameters for.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n_biomarker_total, n_covariate_total) containing the
        generated covariate parameters. Each parameter is drawn from a uniform distribution between 0.3
        and 0.8. Some parameters may be negated with a 50% probability.
    """
    params_cov = []
    for _ in range(n_covariate_total):
        tmp_parms_cov = rng.uniform(0.3, 0.8, (n_biomarker_total, 1))
        tmp_parms_cov = tmp_parms_cov if rng.choice([True, False]) else -tmp_parms_cov
        params_cov.append(tmp_parms_cov)

    return np.hstack(params_cov)


def model_sigmoid(
    t: np.ndarray,
    cov: np.ndarray,
    params: pd.DataFrame,
    rng: np.random.Generator = None,
):
    """
    Calculate the output of a sigmoidal model.

    Args:
        t (np.ndarray): Input time points.
        cov (np.ndarray): Covariate values.
        params (pandas.DataFrame): DataFrame containing model parameters.
            Should have columns 'a', 'b', and covariate-specific columns prefixed with 'Covariate'.
        rng (np.random.Generator, optional): Random number generator for introducing randomness into the model. Default is None.

    Returns:
        np.ndarray: Output of the sigmoidal model for the given inputs.
    """
    covval = np.exp(params.filter(like="Covariate")).values.reshape(1, -1) ** cov
    covval = np.prod(covval, axis=1)
    li = params["a"] + params["b"] * covval * t
    if rng is None:
        output = 1 / (1 + np.exp(-li))
    else:
        output = 1 / (1 + np.exp(-li - rng.normal(0, 0.3, li.shape)))
    return output


def gompertz_surv_inv(s, params):
    """
    Calculate the inverse survival function (quantile function) of the Gompertz distribution.

    Args:
    - s (numpy.ndarray or float): Survival probabilities. It should be in the range (0, 1).
    - params (dict): Parameters of the Gompertz distribution. It should contain the following keys:
        - "c_" (float): Shape parameter. It must be positive.
        - "lambda_" (float): Scale parameter. It must be positive.

    Returns:
    - t (numpy.ndarray or float): Inverse survival function values corresponding to the given survival probabilities.
    """
    t = np.log(1 - params["c_"] / params["lambda_"] * np.log(s)) / params["c_"]
    return t


def data_synthesis(
    rng_seed: int,
    n_biomarker_total: int,
    n_biomarker_noise: int,
    n_covariate_total: int,
    n_covariate_noise: int,
    n_subject: int,
    onset_duration: float,
    observation_period: float,
    surv_inv: Callable[[float, dict], float] = gompertz_surv_inv,
    params_surv: dict = {"c_": 0.25, "lambda_": 0.015},
    left_truncate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthesize simulated data for survival analysis studies.

    Args:
        rng_seed (int): Seed value for the random number generator.
        n_biomarker_total (int): Total number of biomarkers.
        n_biomarker_noise (int): Number of biomarkers to add noise to.
        n_covariate_total (int): Total number of covariates.
        n_covariate_noise (int): Number of covariates to add noise to.
        n_subject (int): Number of subjects.
        onset_duration (float): Duration of the onset period.
        observation_period (float): Duration of the observation period.
        surv_inv (Callable): Inverse survival function.
        params_surv (dict): DataFrame containing survival parameters.
        left_truncate (bool): Remove subjects corresponding to left-truncated.

    Returns:
        tuple: A tuple containing two DataFrames: the first DataFrame represents the synthesized
        data, and the second DataFrame contains the synthesized parameters.
    """
    rng = np.random.default_rng(rng_seed)

    # survival time synthesis
    recuit_time = rng.uniform(0, onset_duration, n_subject)
    event_time = surv_inv(rng.uniform(0, 1, n_subject), params_surv)

    left_t = event_time < recuit_time
    right_c = (recuit_time + observation_period) < event_time
    event = np.logical_not(left_t | right_c)

    survival_time = np.where(right_c, recuit_time + observation_period, event_time)
    followup_time = survival_time - recuit_time

    disease_range = np.ceil(max(survival_time))

    # parameters synthesis
    params_b = rng.uniform(-0.5, 0.5, n_biomarker_total)
    params_x = rng.uniform(0, disease_range, n_biomarker_total)
    params_a = -params_x * params_b

    params_cov = generate_params_cov(rng, n_biomarker_total, n_covariate_total)
    params_cov[:, n_covariate_total - n_covariate_noise :] = 0

    params_output = pd.DataFrame(
        np.column_stack([params_a, params_b, params_cov]),
        columns=["a", "b"]
        + ["Covariate" + str(i + 1) for i in range(n_covariate_total)],
    )

    is_noise_biomarker = rng.choice(
        range(n_biomarker_total), n_biomarker_noise, replace=False, shuffle=False
    )
    params_output.loc[is_noise_biomarker, :] = 0

    # observations synthesis
    time_on_study = np.tile(np.arange(0, observation_period + 1), n_subject)
    offsetT_true = np.repeat(recuit_time, observation_period + 1)
    disease_stage = time_on_study + offsetT_true

    covariates = rng.binomial(1, 0.7, size=(n_subject, n_covariate_total))
    covariates = np.repeat(covariates, observation_period + 1, axis=0)

    biomarkers = np.empty((n_subject * (observation_period + 1), n_biomarker_total))
    for i, val in params_output.iterrows():
        biomarkers[:, i] = model_sigmoid(disease_stage, covariates, val, rng)
    biomarkers = np.where(rng.uniform(size=biomarkers.shape) > 0.7, np.nan, biomarkers)

    df_info = pd.DataFrame(
        dict(
            ID=np.repeat(np.arange(n_subject) + 1, observation_period + 1),
            TIME=time_on_study.astype(np.float32),
            offsetT_true=offsetT_true,
            FUTIME=np.repeat(followup_time, observation_period + 1),
            event_time=np.repeat(event_time, observation_period + 1),
            left_t=np.repeat(left_t, observation_period + 1) * 1,
            event=np.repeat(event, observation_period + 1) * 1,
        )
    )
    df_biomarkers = pd.DataFrame(
        biomarkers, columns=["Biomarker" + str(i + 1) for i in range(n_biomarker_total)]
    )
    df_covariates = pd.DataFrame(
        covariates, columns=["Covariate" + str(i + 1) for i in range(n_covariate_total)]
    )

    df = pd.concat([df_info, df_biomarkers, df_covariates], axis=1)

    if left_truncate:
        df = df.query("left_t == 0").drop("left_t", axis=1).reset_index(drop=True)

    return df, params_output
