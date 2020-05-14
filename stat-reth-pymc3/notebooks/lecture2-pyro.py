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
import seaborn as sns
import altair 
import torch
import pyro
import pyro.distributions as dist
import pandas as pd

from rethinking import MAP, precis

# %% [markdown]
# # Code 2.1
#

# %%
ways = torch.tensor([0., 3, 8, 9, 0])
ways / ways.sum()

# %% [markdown]
# # Code 2.2

# %%
dist.Binomial(total_count=9, probs = 0.5).log_prob(torch.tensor(6.)).exp()

# %% [markdown]
# ## Code 2.3

# %%
size_gride = 20
n = 9
w = 6

# define_grid
p_grid = torch.linspace(start=0, end=1, steps=size_gride)

# define prior
prop2prior = torch.tensor(1.).repeat(size_gride)
prior = prop2prior / torch.sum(prop2prior)

# compute likelihood at each value in grid
likelihood = dist.Binomial(total_count = n, probs = p_grid).log_prob(torch.tensor(6.)).exp()

# compute product of likelihood and prior
unstd_posterior = likelihood * prior

# standarize the posterior, so it sums to 1
posterior = unstd_posterior / unstd_posterior.sum()


# %% [markdown]
# # Code 2.4

# %%
ax = sns.lineplot(p_grid, posterior, marker = 'o')
ax.set(
    xlabel= 'probability of water',
    ylabel = 'posterior probability',
    title = '20 points'
)


# %%
aux = pd.DataFrame({
        
        'p_grid':p_grid,
        'posterior': posterior,
        'prior': prior
        
})

altair_posterior = altair.Chart(aux).mark_line().encode(
    altair.X('p_grid', title='random variable'),
    altair.Y('posterior', title = 'probability')
)

altair_prior =  altair.Chart(aux).mark_line().encode(
    altair.X('p_grid', title='random variable'),
    altair.Y('prior', title = 'probability')
)

altair_prior + altair_posterior

# %%
aux_melt = aux.melt(id_vars = ['p_grid'])
altair.Chart(aux_melt).mark_line().encode(
    x= 'p_grid',
    y= 'value',
    color= 'variable'

)

# %% [markdown]
# # Code 2.5

# %%
prior_unstd = torch.where(p_grid < 0.5, torch.tensor(0.), torch.tensor(1.))
prior = prior_unstd / prior_unstd.sum()


# %%
prior = (-5 * (p_grid - 0.5).abs()).exp()

# %%
sns.lineplot(x = p_grid, y = prior)

# %%
altair.Chart(
    
    pd.DataFrame(
    dict(
        parameter = p_grid,
        prior = prior)
    )

).mark_line().encode(
    
    x = 'parameter',
    y = 'prior'

)


# %% [markdown]
# # Code 2.6

# %% [markdown]
# ## Probabilistic model

# %%
def model(w):
    p = pyro.sample("p", dist.Uniform(0, 1))
    pyro.sample('w', dist.Binomial(9, p), obs=w)

globe_ga = MAP(model).run(torch.tensor(6.))


# display summar of quadratic approximation
precis(globe_ga)

# %%
precis(globe_ga)['Mean'].values

# %% [markdown]
# ## Code 2.7
# ## comparison

# %%
w = 6
n = 9
x = torch.linspace(0, 1, 101)
beta_posterior = dist.Beta(w + 1, n - w + 1).log_prob(x).exp()

mean = precis(globe_ga)['Mean'].values[0]
sdt = precis(globe_ga)['StdDev'].values[0]

quadratic_posterior = dist.Normal(loc=mean, scale=sdt).log_prob(x).exp()



aux_melt = pd.DataFrame({
    'x' : x,
    'beta_posterior': beta_posterior,
    'quadratic_posterior': quadratic_posterior
}).melt(id_vars=['x'])


altair.Chart(aux).mark_line().encode(
    
    x = 'x',
    y = 'value',
    color = 'variable'
    

)




