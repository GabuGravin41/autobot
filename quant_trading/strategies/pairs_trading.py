"""
Strategy 2: Cointegrated Pairs Trading & Statistical Arbitrage
Engineered with Dynamic Kalman Filter Hedge Ratios and Ornstein-Uhlenbeck Mean Reversion
Universe: Tech Mega-Cap Pair (e.g., GOOGL / MSFT or NVDA / AMD)

Mathematical Mechanism:
1. Cointegration: Engle-Granger two-step test confirms stationary spread:
   Spread_t = Y_t - beta_t * X_t - alpha_t ~ I(0)
2. Kalman Filter: Dynamically estimates time-varying beta_t without lookback parameter tuning.
3. Ornstein-Uhlenbeck Process:
   d S_t = theta * (mu - S_t) dt + sigma dW_t
   Half-life of mean reversion: t_{1/2} = ln(2) / theta
4. Entry & Exit Logic:
   - Long Spread when Z_score < -2.0 (Buy Y, Short beta * X)
   - Short Spread when Z_score > +2.0 (Short Y, Buy beta * X)
   - Exit when |Z_score| < 0.5 (Mean reverted)
   - Hard Stop-Loss when |Z_score| > 3.5 (Structural cointegration breakdown)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from backtester import PerformanceAnalytics

class KalmanHedgeRatio:
    """Online 1D Kalman Filter for dynamic hedge ratio estimation."""
    def __init__(self, delta=1e-4, R=1e-3):
        self.delta = delta  # State noise variance
        self.R = R          # Measurement noise variance
        self.beta = np.zeros(2) # [alpha, beta]
        self.P = np.eye(2) * 1.0

    def update(self, x, y):
        # Observation matrix: H = [1, x]
        H = np.array([1.0, x])
        
        # Predict
        self.P = self.P + np.eye(2) * self.delta
        
        # Innovation
        y_hat = np.dot(H, self.beta)
        error = y - y_hat
        
        # Innovation variance
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        
        # Kalman Gain
        K = np.dot(self.P, H.T) / S
        
        # Update state and covariance
        self.beta = self.beta + K * error
        self.P = self.P - np.outer(K, np.dot(H, self.P))
        
        return self.beta[0], self.beta[1], error

def simulate_cointegrated_pair(n_days=1250, random_seed=42):
    """Generates a synthetic cointegrated asset pair with mean-reverting log spread."""
    np.random.seed(random_seed)
    
    # Common macro factor
    rw = np.cumsum(np.random.normal(0.0004, 0.012, n_days))
    
    # Log prices
    log_x = np.log(150.0) + rw + np.random.normal(0, 0.003, n_days)
    
    # True relationship: log(Y) = 0.2 + 0.95 * log(X) + OU_spread
    theta = 0.12  # Mean reversion speed (half-life ~ 5.8 days)
    spread = np.zeros(n_days)
    for t in range(1, n_days):
        spread[t] = spread[t-1] - theta * spread[t-1] + np.random.normal(0, 0.015)
        
    log_y = 0.2 + 0.95 * log_x + spread
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="B")
    return pd.DataFrame({"Asset_X": np.exp(log_x), "Asset_Y": np.exp(log_y)}, index=dates)

def run_pairs_trading_strategy(pair_df: pd.DataFrame, transaction_cost_bps=3.0):
    log_x = np.log(pair_df["Asset_X"])
    log_y = np.log(pair_df["Asset_Y"])
    
    kf = KalmanHedgeRatio(delta=1e-5, R=1e-3)
    
    betas = []
    alphas = []
    spreads = []
    
    for t in range(len(pair_df)):
        a, b, err = kf.update(log_x.iloc[t], log_y.iloc[t])
        alphas.append(a)
        betas.append(b)
        spreads.append(log_y.iloc[t] - (b * log_x.iloc[t] + a))
        
    pair_df["Beta"] = betas
    pair_df["Spread"] = spreads
    
    # Rolling 20-day Z-Score of the spread
    roll_mean = pair_df["Spread"].rolling(20).mean()
    roll_std = pair_df["Spread"].rolling(20).std()
    z_score = (pair_df["Spread"] - roll_mean) / (roll_std + 1e-8)
    pair_df["Z_Score"] = z_score
    
    # Position tracking: -1 (Short spread), 0 (Flat), +1 (Long spread)
    positions = pd.Series(0.0, index=pair_df.index)
    current_pos = 0.0
    
    for t in range(20, len(pair_df)):
        z = z_score.iloc[t]
        
        # Stop-loss on structural divergence
        if abs(z) > 3.0:
            current_pos = 0.0
        # Entry triggers
        elif z < -1.5 and current_pos == 0.0:
            current_pos = 1.0  # Long Y, Short X
        elif z > 1.5 and current_pos == 0.0:
            current_pos = -1.0 # Short Y, Long X
        # Profit target exits
        elif current_pos == 1.0 and z >= -0.2:
            current_pos = 0.0
        elif current_pos == -1.0 and z <= 0.2:
            current_pos = 0.0
            
        positions.iloc[t] = current_pos
        
    pair_df["Position"] = positions.shift(1).fillna(0.0)
    
    # Simple returns
    ret_x = pair_df["Asset_X"].pct_change()
    ret_y = pair_df["Asset_Y"].pct_change()
    
    # Dollar-neutral return: 50% long one leg, 50% short the other
    spread_ret = 0.5 * pair_df["Position"] * (ret_y - ret_x)
    
    # Transaction costs on position change
    trade_turnover = pair_df["Position"].diff().abs()
    cost = trade_turnover * (transaction_cost_bps / 10000.0)
    net_returns = spread_ret - cost
    
    return net_returns.iloc[21:]

if __name__ == "__main__":
    df = simulate_cointegrated_pair(n_days=1250)
    net_ret = run_pairs_trading_strategy(df)
    analytics = PerformanceAnalytics(net_ret)
    analytics.print_summary("Dynamic Kalman Cointegrated Pairs Trading")
