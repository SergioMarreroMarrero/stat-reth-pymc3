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
import pandas as pd
import seaborn as sns
import torch
import pyro
import pyro.distributions as dist
from rethinking import LM, coef

# %% [markdown]
# ## Code 0.1

# %%
print('All models are wrong but some are usefull')

# %% [markdown]
# ## Code 0.2

# %%
x = torch.arange(1., 3)
x = x * 10
x = x.log()
x = x.sum()
x = x.exp()
x

# %% [markdown]
# # Code 0.3

# %%
print(torch.tensor(0.01).pow(200).log())
print(200 * torch.tensor(0.01).log())


# %% [markdown]
# # Code 0.4

# %%
# Load the data:
# car braking distances in feet paired with speeds in km/h
# see cars.infor() for details


# %%
cars = pd.read_csv("../data/cars.csv")
# fit a linear regression of distance on speed
m = LM("dist ~ speed", data=cars).run()
# estimated coefficienes from the model
print(coef(m))

# %%
# plot residuals against speed
y = coef(m)['Intercept'].item() + coef(m)['speed'].item() * cars['speed']
resid = cars['dist'] - y
ax = sns.scatterplot(cars['speed'], resid)
ax.set(xlabel='speed', ylabel='resid')

# %%
import altair as alt

# %%
type(resid)

# %%
aux= pd.DataFrame(
    {
        'x' : range(len(resid.values)),
        'y' : resid.values
    }
)

alt.Chart(aux)\
    .mark_point()\
    .encode(
        x='x',
        y='y'
)
