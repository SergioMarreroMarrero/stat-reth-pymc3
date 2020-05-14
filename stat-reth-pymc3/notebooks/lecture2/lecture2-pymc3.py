# -*- coding: utf-8 -*-
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
import numpy as np
import pandas as pd
import pymc3 as pm
from random import choices
from scipy import stats
import altair as alt


# %%
n=9
p=0.5
k=6

# %% [markdown]
# # Generación de variables aleatorias con:

# %% [markdown]
# **Numpy**

# %%
np.random.binomial(1, p, n)

# %% [markdown]
# **Scipy**

# %%
stats.binom.rvs(1, p, size=n)

# %% [markdown]
# # Funcion de probabilidad de una binomial

# %%
round(stats.binom.pmf(k, n, p), 2)

# %% [markdown]
# Ejemplo:
#

# %%
p_grid = np.linspace(0, 1, 101)
prob_p = np.ones(101)/np.sum(101)
prob_data = stats.binom.pmf(k, n, p=p_grid)
prop2posterior = prob_data * prob_p
p_data = sum(prop2posterior)
posterior = prop2posterior/p_data


# %% [markdown]
# **Veamos la posteriori**

# %%
aux = pd.DataFrame(posterior, columns=['prob']).reset_index()
aux['p'] = aux['index']/100

alt.Chart(aux)\
    .mark_line()\
    .encode(
        x = alt.X('p', title='p'),
        y = alt.Y('prob', title='density')
)



# %% [markdown]
# **Ahora vamos a muestrear de la posteriori**

# %%
samples = pd.DataFrame(
    np.random.choice(p_grid, 5000, p = posterior),
    columns = ['prob']
).reset_index()

samples.head()



# %%
plot_1 = alt.Chart(samples)\
    .mark_point()\
    .encode(
        x = alt.X('index', title='samples'),
        y = alt.Y('prob', title='parameter p of the posterior')
    )

plot_2 = alt.Chart(samples)\
    .mark_area(opacity=0.3, interpolate='step')\
    .encode(
        alt.X('prob:Q', 
              bin=alt.Bin(maxbins=200),
              scale=alt.Scale(domain=(0, 1)),
              title='parameter p of the posterior'
             ),
        alt.Y('count()',
              stack=None,
              title='Number of records')
)
alt.hconcat(plot_1, plot_2)

# %% [markdown]
# El gráfico de la derecha muesta la distribución de los puntos del scatter-plot de la izquierda

# %% [markdown]
# ## Intervalos de credabilidad

# %% [markdown]
# 1. Numpy 

# %% [markdown]
# El parámetro p se encuentra en el intervalo [0.35, 0.88] con un 95% de probabilidad

# %%
list(map(lambda x: np.round(x, 2),
         [np.percentile(np.array(samples['prob']), 2.5), np.percentile(np.array(samples['prob']), 97.5)]))


# %% [markdown]
# 2. Pymc3 

# %%
pm.stats.quantiles(
    np.array(samples.prob),
    qlist=[2.5, 97.5]
)


# %% [markdown]
# ## Ejercicio 1

# %% [markdown]
# Suppose the globe tossing data had turned out to be 8 water in 15 tosses. Construct the posterior distribution, using grid approximation. Tuse the same flat prior as before

# %%
k=8
n=15

# %%
p_grid = np.linspace(0, 1, 101)
likelihood = stats.binom.pmf(k, n, p=p_grid)
prop2prior = np.full(101, 1)
prior = prop2prior / np.sum(prop2prior)

prop2posterior = likelihood * prior
posterior = prop2posterior / np.sum(prop2posterior)

np.sum(posterior)

# %%
p_grid = np.linspace(0, 1, 101)
prob_p = np.concatenate([
    np.zeros(50),
    np.full(51, 0.5)
    ])

prob_p = prob_p / np.sum(prob_p)
prob_data = stats.binom.pmf(k, n , p=p_grid)
prop2posterior2 = prob_data * prob_p
posterior2 = prop2posterior / sum(prop2posterior)


samples = pd.DataFrame({'prob': np.random.choice(p_grid, 5000, p=posterior2)})


# %%
list(map(lambda x: np.round(x, 2),
         [sum(posterior2 * p_grid), np.mean(samples['prob'])]))

# %%
pm.stats.quantiles(np.array(samples.prob), qlist=[0.05, 99.5])

# %% [markdown]
# ## Exercise 2
# p < 0.5 is zero and p > 0.5 is constant

# %%
dfe2 = pd.DataFrame(
    
    {
    'p_grid': np.linspace(0, 1, 101),
    'prob_p': np.concatenate((np.zeros(50), np.ones(51)/51))
    }

)

dfe2['likelihood'] = stats.binom.pmf(k, n, p_grid)
dfe2['prop2post'] = dfe2['likelihood'] * dfe2['prob_p']


# %%
dfe2['posterior'] = dfe2['prop2post'] / np.sum(dfe2['prop2post'])

dfe2_melt = dfe2[['p_grid', 'prob_p', 'posterior']].melt(id_vars = 'p_grid')
dfe2_melt.head()


# %%
probs = alt.Chart(dfe2_melt)\
    .mark_line()\
    .encode(
    
    x = 'p_grid',
    y = 'value',
    color = 'variable'

)

rule = alt.Chart(dfe2_melt).mark_rule(color='red')\
    .encode(
    x = 'a:Q',
    size = alt.value(2)

)

(probs + rule).transform_calculate(a='0.7')



# %% [markdown]
#
# ## Exercise 3
# Suppose we need to give the result with a ceartin precion.

# %%

def binomial_model(n, p=0.7):
    
    k = np.sum(np.random.binomial(1, p, n))
    df = pd.DataFrame(

        {
        'p_grid': np.linspace(0, 1, 101),
        'prior': np.concatenate((np.zeros(50), np.ones(51)/51))
        }
    )
        
    df['likelihood'] = stats.binom.pmf(k, n, df['p_grid'])
    df['prop2post'] = df['likelihood'] * df['prior'] 
    df['posterior'] = df['prop2post'] / np.sum(df['prop2post'])
    
    posterior_samples = np.random.choice(df['p_grid'], p=df['posterior'], size=5000)
    quantiles = pm.stats.quantiles(posterior_samples, qlist=[0.05, 99.5])

    return quantiles[99.5] - quantiles[0.05] 

mysims = {}
for n in [50, 100, 200, 500, 1000, 2000, 5000]:
    print(n)
    nsim=500
    sim_sample = [binomial_model(n = n) for sim in range(nsim)]
    
    
    mysims[n] = {
            
        'mean': np.mean(sim_sample),
        'sdt_error': np.std(sim_sample)/np.sqrt(nsim) 
    }
        
    
import json
print(json.dumps(mysims, indent=5))
    
    
    



