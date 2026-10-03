import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

DATA_DIR = Path("competitions/soil_grain_size_photos/data")
train_labels = pd.read_csv(DATA_DIR / "Training_labels_updated.csv")
target_cols = [c for c in train_labels.columns if c != "sample_id"]
diameters = np.array([float(c) for c in target_cols])
targets = train_labels[target_cols].values

def weibull_cdf(d, d0, n):
    return 100.0 * (1.0 - np.exp(- (d / d0) ** n))

def emd_score(y_true, y_pred):
    log_d = np.log10(diameters)
    weights = np.diff(log_d)
    diff = np.abs(y_true - y_pred)
    trapezoid = 0.5 * (diff[:, :-1] + diff[:, 1:]) * weights
    return trapezoid.sum(axis=1).mean()

# Fit Weibull to each training sample to measure the approximation error
weibull_approx = []
fit_params = []
for i in range(len(targets)):
    y = targets[i]
    # initial guess
    d50_idx = np.searchsorted(y, 50.0)
    d0_init = diameters[min(d50_idx, len(diameters)-1)]
    popt, _ = curve_fit(weibull_cdf, diameters, y, p0=[d0_init, 1.0], bounds=([1e-4, 0.1], [100.0, 10.0]))
    fit_params.append(popt)
    weibull_approx.append(weibull_cdf(diameters, *popt))

weibull_approx = np.array(weibull_approx)
fit_error = emd_score(targets, weibull_approx)
print(f"Theoretical Lower Bound of 2-parameter Weibull hypothesis on training set: EMD = {fit_error:.3f}")

# Also compare with Sqrt-Mass reconstruction
mass = np.diff(np.column_stack([np.zeros(len(targets)), targets]), axis=1)
sqrt_mass = np.sqrt(mass / 100.0)
recon_mass = (sqrt_mass ** 2) * 100.0
recon_cdf = np.cumsum(recon_mass, axis=1)
print(f"Exact Sqrt-Mass reconstruction error: EMD = {emd_score(targets, recon_cdf):.6f}")
