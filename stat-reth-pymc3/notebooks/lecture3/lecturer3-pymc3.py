# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pymc3 as pm
import numpy as np
import pandas as pd
import theano

from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler 
import math

import matplotlib.pyplot as plt
import altair as alt
alt.data_transformers.enable('default', max_rows=None)
import arviz as az

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Data Cleaning

# %%
# Train Data
X_train = pd.read_csv('../../data/Howell1.csv', sep=';', header=0)
# Only adults
X_train = X_train[X_train['age'] >= 18].reset_index(drop=True)
# Scale
scaler = StandardScaler(with_mean=True, with_std=False)
X_train['weight_c'] = scaler.fit_transform(X_train[['weight']])


# Predict Data
X_results = pd.DataFrame({
    
    'Individual': range(5),
    'weight': [45, 40, 65, 31, 53],
    'expected height' : np.nan,
    '89% interval' : np.nan
    
    
})

X_results['weight_c'] = scaler.transform(X_results[['weight']])




# %% [markdown]
# ## Model Definition
#

# %%
with pm.Model() as model:
    
    # Data
    weight = pm.Data('weight', X_train['weight'].values)
    height = pm.Data('height', X_train['height'].values)
    
    # Priors
    alpha = pm.Normal('alpha', mu=178, sd=20)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    
    # Regression
    mu = alpha + beta * weight
    height_hat = pm.Normal('height_hat', mu=mu, sd=sigma, observed=height)
    
    # Prior sampling, trace definition and posterior sampling
    prior = pm.sample_prior_predictive()
    posterior = pm.sample(draws=1000, tune=1000)
    posterior_pred = pm.sample_posterior_predictive(posterior)




# %% [markdown]
# ## Checking chain

# %%
# Print posterior summary     
print(az.summary(posterior, credible_interval=0.80)) 
az.plot_trace(posterior)

# %% [markdown]
# ### Make predictions on train

# %%
X_train['height_hat'] = np.mean(posterior_pred['height_hat'], axis=0)
height_hat_hpd = pm.hpd(posterior_pred['height_hat'], credible_interval=0.9).round(2)
X_train['lower_hdpi'] = height_hat_hpd[:, 0]
X_train['upper_hdpi'] = height_hat_hpd[:, 1]



# %% [markdown]
# ### RMSE (Check model performance)

# %%
RMSE = np.sqrt(mean_squared_error(X_train['height'], X_train['height_hat']))


# %% [markdown]
# # Make predictions on test data

# %%
weight.set_value(X_results['weight_c'].values)


# %%
posterior_pred_test = pm.sample_posterior_predictive(
    trace = posterior,
    samples = 500,
    model = model

)

# %%
X_results['expected height'] = posterior_pred_test['height_hat'].mean(axis=0)
X_results['89% interval'] = list(pm.hpd(posterior_pred_test['height_hat'], credible_interval=0.89).round(2))
X_results

# %%
