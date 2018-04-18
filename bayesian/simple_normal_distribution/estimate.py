import numpy as np
import pymc as pm
import pprint as pp

mean_true = 5
variance_true = 2

data = np.random.normal(mean_true, variance_true, 1000)

mean = pm.Uniform('mean', lower=min(data), upper=max(data))
precision = pm.Uniform('precision', lower=0.0001, upper=1)

likelihood = pm.Normal('likelihood', mu=mean, tau=precision, observed=True, value=data)

model = pm.Model([mean, precision, likelihood])

M = pm.MCMC(model)
M.sample(iter = 40000, burn=5000)

pp.pprint(M.stats())
