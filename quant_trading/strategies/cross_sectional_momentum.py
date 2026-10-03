"""
Strategy 1: Cross-Sectional Dual-Momentum with Volatility-Inversion Risk Budgeting
Universe: Multi-Asset ETF Basket (SPY, QQQ, TLT, GLD, IWM)
Mechanism:
- Absolute Trend Filter: Asset must be above its 100-day Simple Moving Average (SMA).
- Relative Strength: Rank assets by 20-day exponential momentum score.
- Inverse-Volatility Weighting: Allocate capital inversely proportional to 30-day realized volatility:
  w_i = (1 / sigma_i) / sum(1 / sigma_j)
- Cash Regime Filter: When composite equity market momentum is negative, shift allocation to Short-Term Treasuries (Cash).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from backtester import PerformanceAnalytics

def simulate_multi_asset_history(n_days=1000, random_seed=42):
    """Generates realistic correlated geometric random walks for backtesting."""
    np.random.seed(random_seed)
    tickers = ["SPY", "QQQ", "TLT", "GLD", "IWM"]
    
    # Annualized drift and volatility
    mu = np.array([0.10, 0.14, 0.04, 0.07, 0.09]) / 252.0
    sigma = np.array([0.16, 0.22, 0.14, 0.15, 0.20]) / np.sqrt(252.0)
    
    # Correlation matrix
    corr = np.array([
        [ 1.00,  0.88, -0.35,  0.05,  0.82],
        [ 0.88,  1.00, -0.38,  0.02,  0.80],
        [-0.35, -0.38,  1.00,  0.25, -0.32],
        [ 0.05,  0.02,  0.25,  1.00,  0.08],
        [ 0.82,  0.80, -0.32,  0.08,  1.00]
    ])
    
    L = np.linalg.cholesky(corr)
    Z = np.random.normal(size=(n_days, len(tickers)))
    correlated_returns = mu + np.dot(Z, L.T) * sigma
    
    # Prices
    prices = 100.0 * np.exp(np.cumsum(correlated_returns, axis=0))
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates, columns=tickers)

def run_momentum_strategy(prices_df: pd.DataFrame, transaction_cost_bps=3.0):
    returns_df = prices_df.pct_change()
    
    # 1. Signals: 20-day momentum and 100-day trend filter
    mom_20 = prices_df.pct_change(20)
    sma_100 = prices_df.rolling(100).mean()
    trend_filter = prices_df > sma_100
    
    # 2. Realized 30-day volatility
    vol_30 = returns_df.rolling(30).std()
    
    # 3. Strategy Weights
    weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    
    for t in range(100, len(prices_df)):
        current_date = prices_df.index[t]
        moms = mom_20.iloc[t]
        valid = trend_filter.iloc[t] & (moms > 0)
        
        if valid.sum() > 0:
            # Select top 2 performing assets that pass trend filter
            eligible = moms[valid].nlargest(2).index
            vols = vol_30.iloc[t][eligible]
            inv_vols = 1.0 / (vols + 1e-8)
            norm_weights = inv_vols / inv_vols.sum()
            weights.loc[current_date, eligible] = norm_weights
        else:
            # Cash / Treasury allocation (0% equity exposure)
            pass
            
    # Shift weights by 1 day to prevent lookahead bias
    exec_weights = weights.shift(1).fillna(0.0)
    
    # Daily portfolio returns
    gross_returns = (exec_weights * returns_df).sum(axis=1)
    
    # Deduct transaction costs on rebalancing
    turnover = exec_weights.diff().abs().sum(axis=1)
    cost = turnover * (transaction_cost_bps / 10000.0)
    net_returns = gross_returns - cost
    
    return net_returns.iloc[101:]

if __name__ == "__main__":
    prices = simulate_multi_asset_history(n_days=1250)
    strat_ret = run_momentum_strategy(prices)
    analytics = PerformanceAnalytics(strat_ret)
    analytics.print_summary("Cross-Sectional Dual Momentum + Volatility Budgeting")
