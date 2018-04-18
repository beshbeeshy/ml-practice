import numpy as np
import pymc as pm
import pprint as pp
import matplotlib.pyplot as plt

mean_true = 5
variance_true = 4

data = np.random.normal(mean_true, variance_true, 1000)

mean = pm.Uniform('mean', lower=min(data), upper=max(data))
precision = pm.Uniform('precision', lower=0.0001, upper=1)

likelihood = pm.Normal('likelihood', mu=mean, tau=precision, observed=True, value=data)

model = pm.Model([mean, precision, likelihood])

M = pm.MCMC(model)
M.sample(iter = 50000, burn=5000)

mean_samples = M.trace('mean')[:]
precision_samples = M.trace('precision')[:]

ax = plt.subplot(211)
plt.hist(mean_samples, histtype='stepfilled', bins=50, alpha=0.85,
         label="mean samples", color="#A60628", normed=True)

ax = plt.subplot(212)
plt.hist(precision_samples, histtype='stepfilled', bins=50, alpha=0.85,
         label="precision samples", color="#7A68A6", normed=True)

plt.show()
