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

# %% [markdown]
#
# ## Exercise 2
#
# Model the relationship between height (cm) and the natural logarithm of weight (log-kg). Don't filter only with the adults. Use any model type from Chapter 4 that you thinki is useful: an ordinary linear regression, polynomial or spline. Plot the posterior prediction against the raw data.

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

# %%
# Train Data
X_train = pd.read_csv('../../data/Howell1.csv', sep=';', header=0)

# Take log
X_train['log_weight'] = np.log(X_train['weight'])

# scale
scaler = StandardScaler(with_mean=True)
X_train
X_train['weight_c'] = scaler.fit_transform(X_train[['weight']]) 



#Log Scale
log_scaler = StandardScaler(with_mean=True, with_std=False)
X_train['log_weight_c'] = log_scaler.fit_transform(X_train[['log_weight']])




nolog_relation = alt.Chart(X_train).mark_circle().encode(
    x = 'weight_c',
    y = 'height'
)

log_relation = alt.Chart(X_train).mark_circle().encode(
    x = 'log_weight_c',
    y = 'height'
)

nolog_relation | log_relation

# %%
# Predict Data
X_results = pd.DataFrame({
    
    'Individual': range(5),
    'weight': [45, 40, 65, 31, 53],
    'expected height' : np.nan,
    '89% interval' : np.nan
    
    
})

X_results['weight_c'] = scaler.transform(X_results[['weight']])

# %% [markdown]
# ## Model 

# %%
with pm.Model() as model:
    
    # Data
    log_weight_c = pm.Data('log_weight_c', X_train['log_weight_c'])
    height = pm.Data('height', X_train['height'])
    
    # Priors
    alpha = pm.Normal('alpha', mu=178, sd=20)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    
        # Equations
    mu = alpha + beta * log_weight_c
    height_hat = pm.Normal('height_hat', mu=mu, sd=sigma, observed=height)
    
    
    # Samples
    prior = pm.sample_prior_predictive()
    posterior = pm.sample(draws=1000, tune=1000)
    posterior_pred = pm.sample_posterior_predictive(posterior)


# %%
print(az.summary(posterior, credible_interval=.89).round(2))
az.plot_trace(posterior)

# %%
X_train['height_hat'] = np.mean(posterior_pred['height_hat'], axis=0).round(2)
height_pred_hpd = pm.hpd(posterior_pred['height_hat'], credible_interval=0.89)
X_train['lower_hpid'] = height_pred_hpd[:, 0].round(2)
X_train['upper_hpid'] = height_pred_hpd[:, 1].round(2)



# %%
RMSE = np.sqrt(mean_squared_error(X_train['height'], X_train['height_hat'])).round(2)
RMSE

# %%
X_train

# %%
# melt to plot
X_train_melt = X_train.melt(id_vars = ['weight'],
                           value_vars = ['height', 'height_hat'])


# %%
plot = alt.Chart(X_train_melt)\
    .mark_point()\
    .encode(
        x = alt.X('weight', title='log_weight', scale=alt.Scale(domain=(0, 80))),
        y = alt.Y('value', title='value', scale=alt.Scale(domain=(0, 180))), 
        color='variable'
)

plot_hpdi = alt.Chart(X_train)\
    .mark_area(opacity=0.2, color='orange')\
    .encode(
        x = alt.X('weight', title='weight', scale=alt.Scale(domain=(0, 80))),
        y = 'lower_hpid:Q',
        y2 = 'upper_hpid:Q'
)


plot_2 = alt.Chart(X_train)\
   .mark_point(color='black')\
   .encode(
        x=alt.X('height', title='height',scale=alt.Scale(domain=(0, 180))),
        y=alt.Y('height_hat', title='height_hat',scale=alt.Scale(domain=(0, 180)))
          )

(plot+plot_hpdi) | plot_2
