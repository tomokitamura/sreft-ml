# Welcome to sreft-ml


## What is SReFT-ML?
SReFT-ML is a neural network-based method for constructing long-term disease progression models from short-term observational data. It updates the SReFT algorithm to a machine learning-based approach. For detailed information about the concept of SReFT, please refer to [the original paper](https://ascpt.onlinelibrary.wiley.com/doi/10.1002/cpt.1166).


## How to use sreft-ml
### Required Data
SReFT-ML uses time series data of multiple biomarkers. Each record consists of ID, time, and observed value of each biomarker, as shown in the table below. Please format the data beforehand, since the program is created assuming this format. 
**Note that the column names must be `ID` for ID and `TIME` for time.**

|ID|TIME|Biomarker 1|Biomarker 2|...|
|----:|----:|----:|----:|----|
|1|0|0.1|0.5|...|
|1|1|0.2|0.4|...|
|...|...|..|...|...|
|100|5|1|2.4|...|

### Preprocessing
Use [utilities.split_data_for_sreftml](https://ryotajin.github.io/sreft-ml/reference/#sreftml.utilities.split_data_for_sreftml) to preprocess the data set prepared in the previous section for SReFT-ML.

```python
x, cov, m, y = utilities.split_data_for_sreftml(
    df, name_biomarkers, name_covariates, isMixedlm=isMixedlm
)
```

Standardize the data if necessary. This is a fundamental and indispensable step since in many cases the scale differs between biomarkers. For the scaler, we assume scikit-learn's Scaler. Any Scaler may be used. Choose the one that best suits the nature of your data.
**But do not convert x. x is in pandas format, so please only convert it to numpy format.**

```python
x_scaled = x.values.reshape(-1, 1)
cov_scaled = scaler_cov.fit_transform(cov.values)
m_scaled = scaler_m.fit_transform(m.values)
y_scaled = scaler_y.fit_transform(y.values)
```

### Hyperparameter Search
To search for hyperparameters, use [sreftml_model.hp_search_for_sreftml](https://ryotajin.github.io/sreft-ml/reference/#sreftml.sreftml_model.hp_ search_for_sreftml) to perform random search or grid search.
Currently, only the items specified in `grid_dict` below can be targeted.
The `df_grid` contains the results. The one with the best `score` should be used for model building.

```python
grid_dict = {
    "adam_lr": [1e-3, 1e-4, 1e-5],
    "activation_model_1_mid": ["sigmoid", "tanh"],
    "activation_model_1_out": ["relu", "softplus"],
    "activation_model_y_mid": ["relu", "tanh", "linear"],
}

df_grid = sreftml_model.hp_search_for_sreftml(
    df,
    (x_scaled, cov_scaled, m_scaled, y_scaled),
    grid_dict=grid_dict,
    n_grid_sample=0,
    n_splits=3,
)
```

### Model Building
Split the data into a training set and a validation set set. This is supposed to be done for each ID, so we use `sklearn.model_selection.GroupShuffleSplit`.

```python
from sklearn.model_selection import GroupShuffleSplit
((train_idx, vali_idx),) = GroupShuffleSplit(
    1, test_size=0.1, random_state=random_seed
).split(X=df, groups=df.ID)
```

Build the model. Since `sreftml_model.SReFT` implements TensorFlow's Subclassing API, its usage conforms to it.

```python
sreft = sreftml_model.SReFT(
    output_dim=len(name_biomarkers),
    latent_dim_model_1=m_scaled.shape[1],
    latent_dim_model_y=y_scaled.shape[1],
    activation_model_1_mid="tanh",
    activation_model_1_out="softplus",
    activation_model_y_mid="tanh",
    random_state=random_seed,
)
sreft.compile(optimizer=keras.optimizers.Adam(1e-5))
sreft.fit(
    (
        x_scaled[train_idx, :],
        cov_scaled[train_idx, :],
        m_scaled[train_idx, :],
        y_scaled[train_idx, :],
    ),
    y_scaled[train_idx, :],
    batch_size=256,
    validation_data=(
        (
            x_scaled[vali_idx, :],
            cov_scaled[vali_idx, :],
            m_scaled[vali_idx, :],
            y_scaled[vali_idx, :],
        ),
        y_scaled[vali_idx, :],
    ),
    epochs=9999,
    verbose=0,
)
```

Once you have built the model, calculate and save the predictions, etc. Use in plots.

```python
df = utilities.calculate_offsetT_prediction(
    sreft, df, (x_scaled, cov_scaled, m_scaled, y_scaled), scaler_y, name_biomarkers
)
```

### Creating Plots
Various plotting functions are available. See [help page](https://ryotajin.github.io/sreft-ml/reference/#sreftml.plots) for details.
