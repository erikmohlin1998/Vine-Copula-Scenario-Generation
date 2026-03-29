"""
Vine Copula Scenario Generation

This script models the joint behaviour of multiple asset returns by combining
univariate GJR-GARCH marginal models with a fitted vine copula dependence structure.
Standardized residuals are transformed to the copula scale using the probability
integral transform (PIT), simulated from the fitted copula, and mapped back into
one-step-ahead return scenarios.

The project is intended as a compact quantitative finance example of volatility
modeling, dependence modeling, and multivariate scenario generation.
"""

#%% Relevant libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from arch import arch_model
from arch.univariate import SkewStudent
import yfinance as yf
from scipy.stats import norm, t
import scipy.stats as stats
import pyvinecopulib as pv
import graphviz 
import os
os.environ["PATH"] += os.pathsep + r"C:\Program\Graphviz\bin" # Needed for vine copula tree plots (Windows)

#%%
# import data from yahoo finance
# Get the financial time series for a set of stocks over a given time period. For example:
tickers = ['KO','PEP','MSFT','AAPL']    
start = '2020-01-01'
end = '2024-12-31'
fin_data = yf.download(tickers, start=start, end=end, auto_adjust=False)
print(fin_data.columns)

# Calculate log returns
prices = fin_data['Adj Close']
returns = np.log(prices).diff().dropna()

#%% Marginal distribution modelling of standardised residuals using GJR-GARCH(1,1) with t-distributed errors.
def estimate_garch(returns, dist="t"):
    
    results = {}
    std_residuals = pd.DataFrame(index=returns.index)

    for asset in returns.columns:
        
        model = arch_model(
            returns[asset] * 100,
            mean='Constant',
            vol='GARCH',
            p=1, o=1, q=1,
            dist=dist
        )
        
        res = model.fit(disp="off")
        results[asset] = res
        
        z = res.resid / res.conditional_volatility
        std_residuals[asset] = z

    return results, std_residuals

# Probability integral transform based on the fitted GARCH models, to obtain uniform ~(0,1) data for copula fitting.
def probability_integral_transform(std_residuals, results, dist="t"):
    
    u_data = pd.DataFrame(index=std_residuals.index)
    skewt_dist = SkewStudent()

    for asset in std_residuals.columns:
        
        z = std_residuals[asset]
        
        if dist == "normal":
            u = norm.cdf(z)
            
        elif dist == "t":
            nu = results[asset].params["nu"]
            u = t.cdf(z, df=nu,scale=np.sqrt((nu-2)/nu)) # Rescaling for variance = 1

        elif dist == "skewt":
            eta = results[asset].params["eta"]
            lam = results[asset].params["lambda"]
            u = skewt_dist.cdf(z, parameters=np.array([eta, lam]))  

        else:
            raise ValueError("Distribution not supported.")
        
        u_data[asset] = u

    return u_data


#%% Dist can be changed to "normal" for Gaussian errors, or "skewt" for skewed t-distribution errors
dist = "t"

# Estimate GJR-GARCH models and get standardised residuals for the chose distribution
results, std_residuals = estimate_garch(returns, dist=dist)
# Print the results from each asset
for asset, res in results.items():
    print(f"\n===== {asset} =====")
    print(res.summary())

# Usage of the probability integral transform, change dist as needed
u_data = probability_integral_transform(std_residuals, results, dist=dist)

# Plot the (approximately)uniform data
for asset in u_data:
    plt.hist(u_data[asset], bins=10, density=True, alpha=0.6, color='b')
    plt.title(f"Approximately uniform data after probability integral transform ({asset})")
    plt.show()

# QQ-Plots against uniform
for asset in u_data:
    stats.probplot(u_data[asset],dist="uniform",plot=plt)
    plt.title(f"{asset} PIT QQ Plot")
    plt.show()

# ----------------------------------------------VINE COPULA FITTING AND SIMULATION----------------------------------------------
# %% Family Set notation for pyvinecopulib
# 0 = Independence 
# 1 = Gaussian 
# 2 = Student t 
# 3 = Clayton 
# 4 = Gumbel 
# 5 = Frank 
# 6 = Joe
# 7 = BB1
# 8 = BB6
# 9 = BB7
# 10 = BB8
# 11 = Tawn
# 12 = TLL (non-parametric kernel density estimator, Geenens et al. (2014))

# %% Pairwise plots of the PIT uniform data
fig, axs = pv.pairs_copula_data(u_data)
fig.suptitle("Pairwise dependence (copula scale)", fontsize=25)
tickers = list(u_data.columns)
d = len(tickers)
for i in range(d):
    ax = axs[i, i]
    ax.clear()  # remove the histogram from original pairwise plot
    ax.text(
        0.5, 0.5, tickers[i],
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=20
    )
    ax.set_xticks([])
    ax.set_yticks([])
# Remove x-labels on bottom row and y-labels on left column
d = axs.shape[0]
for j in range(d):
    axs[d-1, j].set_xlabel("")   # bottom row x-labels
for i in range(d):
    axs[i, 0].set_ylabel("")     # left column y-labels
fig.tight_layout(rect=[0, 0, 1, 0.99])

# %% Fitting vine copula to the data using pyvinecopulib. Chooses best fit automatically based on AIC, BIC or log-likelihood (user specified below).
# Controls sets options for fitting. Change families as needed, and whether to automatically select truncation level. 
# num_threads can be set to number of CPU threads to speed things up (check your CPU specs)
controls = pv.FitControlsVinecop(family_set=[1,2,3,4,5,6,7,8,9,10,11,12],select_trunc_lvl=False, num_threads=12,selection_criterion="bic") 
vine_cop =pv.Vinecop.from_data(u_data.to_numpy(),controls=controls) 
print(vine_cop)
vine_cop.plot()

# %% Simulation from fitted vine copula, and same pairwise plot as before
u_sim = vine_cop.simulate(1000,num_threads=12) # Vine copula simulation for all assets, change number of simulations as needed

fig, axs = pv.pairs_copula_data(u_sim)
fig.suptitle("Simulated data from fitted vine copula", fontsize=25)
tickers = list(u_data.columns)
d = len(tickers)
for i in range(d):
    ax = axs[i, i]
    ax.clear()  # remove the histogram
    ax.text(
        0.5, 0.5, tickers[i],
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=20
    )
    ax.set_xticks([])
    ax.set_yticks([])
# Remove x-labels on bottom row and y-labels on left column
d = axs.shape[0]
for j in range(d):
    axs[d-1, j].set_xlabel("")   # bottom row x-labels
for i in range(d):
    axs[i, 0].set_ylabel("")     # left column y-labels
fig.tight_layout(rect=[0, 0, 1, 0.99])

# %% Functions for inverse PIT/One-step return scenarios from std residuals simulations
def inverse_pit_to_std_residuals(u_sim, results, asset_names, dist="t", eps=1e-6, z_cap=12.0):
    U_sim = np.clip(np.asarray(u_sim), eps, 1 - eps)
    Z_sim = np.zeros_like(U_sim, dtype=float)

    skewt_dist = SkewStudent()
    
    for j, asset in enumerate(asset_names):
        pits = U_sim[:,j]

        if dist == "normal":
            z = norm.ppf(pits)
        
        elif dist == "t":
            nu = float(results[asset].params["nu"])
            scale = np.sqrt((nu - 2.0) / nu)  # must match initial PIT scaling
            z = t.ppf(pits, df=nu, scale=scale)

        elif dist =="skewt":
            eta = float(results[asset].params["eta"])
            lam = float(results[asset].params["lambda"])
            z = skewt_dist.ppf(pits, parameters=np.array([eta, lam]))

        else:
            raise ValueError(f"Distribution not supported: {dist}")
        
        Z_sim[:, j] = np.clip(z, -z_cap, z_cap)

    return Z_sim

def one_step_return_scenarios_from_z(results, Z_sim, asset_names, scale_factor=100.0):
    n_samples, d = Z_sim.shape
    R_sim = np.zeros((n_samples, d), dtype=float)

    for j, asset in enumerate(asset_names):
        res = results[asset]
        f = res.forecast(horizon=1, reindex=False)
        mu_f = float(f.mean.values[-1, 0])
        sigma_f = float(np.sqrt(f.variance.values[-1, 0]))
        R_sim[:, j] = (mu_f + sigma_f * Z_sim[:, j]) / scale_factor  # back to decimal returns

    return R_sim

# %% One step ahead return scenarios from the simulated vine copula data, by first transforming back to standardised residuals 
# using the inverse PIT, and then applying the GARCH forecast to get return scenarios.
asset_names = list(u_data.columns)  
Z_sim = inverse_pit_to_std_residuals(u_sim, results, asset_names, dist=dist)

R_sim = one_step_return_scenarios_from_z(results, Z_sim, asset_names)
R_sim = pd.DataFrame(R_sim, columns=asset_names)

print("One-step ahead return scenarios from simulated vine copula data (first five rows):")
print("")
print(R_sim.head())