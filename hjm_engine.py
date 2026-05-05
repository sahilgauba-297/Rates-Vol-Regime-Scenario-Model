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


# monte carlo shocks
shocks = np.random.multivariate_normal([0,0], cov_matrix, size = 100)

#simulating future paths
start_2Y = df["2Y"].iloc[-1]
start_10Y = df["10Y"].iloc[-1]

sim_2Y = [start_2Y]
sim_10Y = [start_10Y]

for i in range(100):

    shock_2Y, shock_10Y = shocks[i]

    new_2Y = sim_2Y[-1]  + shock_2Y
    new_10Y = sim_10Y[-1] + shock_10Y

    sim_2Y.append(new_2Y)
    sim_10Y.append(new_10Y)


print("simulated 2Y path", sim_2Y[:5])
print("simulated 10Y path", sim_10Y[:5])