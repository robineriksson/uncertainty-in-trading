import numpy as np
import polars as pl
from hmmlearn import hmm

from explore_data import load_data

data = load_data()


features = data.select(['returns','ranges'])

model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=1000,
)
model.fit(features)

states = model.predict(features)

import matplotlib.pyplot as plt

plt.hist(states)
plt.show()

tvec = data['date']
plt.plot(tvec, states)
plt.show()

out = data.with_columns(state = states).pivot(index='date',columns='state',values='close')

plt.plot(tvec, out['0'],label='state 0')
plt.plot(tvec, out['1'],label='state 1')
plt.plot(tvec, out['2'],label='state 2')
plt.legend()
plt.show()
