from macro_features import macro_engine
import numpy as np

df = macro_engine()

df["d2Y"] = df["2Y"].diff()
df["d10Y"] = df["10Y"].diff()

df = df.dropna()

vol_2Y = df["d2Y"].std()
vol_10Y = df["d10Y"].std()
corr = df["d2Y"].corr(df["d10Y"])


# building the corelation matrix
cov_matrix = np.array([
[vol_2Y**2, corr * vol_2Y * vol_10Y],
[corr * vol_2Y * vol_10Y, vol_10Y**2]
])

# macro drift

macro_signal = df["macro_score_smooth"].iloc[-1]
drift_2Y = 0.01 * np.tanh(macro_signal)
drift_10Y = 0.8 * drift_2Y


# Simulating paths and steps
n_paths = 1000
n_steps = 100

start_2Y = df["2Y"].iloc[-1]
start_10Y = df["10Y"].iloc[-1]

paths_2Y = np.zeros((n_paths, n_steps))
paths_10Y = np.zeros((n_paths, n_steps))

for p in range(n_paths):
    shocks = np.random.multivariate_normal([0,0], cov_matrix, size = n_steps)

    paths_2Y[p,0] = start_2Y
    paths_10Y[p,0] = start_10Y

    for t in range(1, n_steps):
        shock_2Y, shock_10Y = shocks[t]

        paths_2Y[p, t] = paths_2Y[p, t-1] +shock_2Y + drift_2Y
        paths_10Y[p,t] = paths_10Y[p, t-1] +shock_10Y + drift_10Y



mean_2Y = paths_2Y.mean(axis = 0)
mean_10Y = paths_10Y.mean(axis = 0)

p5_2Y = np.percentile(paths_2Y, 5, axis = 0)
p95_2Y = np.percentile(paths_2Y, 95, axis = 0)

p5_10Y = np.percentile(paths_10Y, 5, axis = 0)
p95_10Y = np.percentile(paths_10Y, 95, axis = 0)


# steepening probability 

initial_spread =  start_10Y  - start_2Y
final_spread = paths_10Y[:, -1] - paths_2Y[:, -1]

steepening_prob = np.mean(final_spread > initial_spread)


print("Steepening probability:", round(steepening_prob, 4))

print("Mean final 2Y:", mean_2Y[-1])
print("Mean final 10Y:", mean_10Y[-1])
