"""
Strategy 3: Crypto Delta-Neutral Funding Rate Cash-and-Carry Arbitrage
Universe: High-Liquidity Perpetual Futures vs. Spot (BTC/USDT, ETH/USDT, SOL/USDT)
Platform Compatibility: CCXT, Binance, Bybit, OKX, Alpaca Crypto

Mathematical Mechanism:
1. Basis & Funding Rate:
   In crypto perpetual swaps, perpetual prices deviate from index spot prices.
   Longs pay shorts when funding rate F_t > 0 (bullish funding premium).
   Shorts pay longs when funding rate F_t < 0.
2. Delta-Neutral Cash-and-Carry:
   - When 8-hour annualized funding rate F_annual > 12.0%:
     * Buy Spot Asset (Delta = +1.0)
     * Short 1x Inverse/Linear Perpetual Futures (Delta = -1.0)
     * Net Portfolio Delta = 0.0 (Zero directional price risk)
     * Collect funding yield every 8 hours (3 times daily).
3. Exit & Reversal Triggers:
   - Exit when annualized funding rate drops below 3.0% (capital reallocation to higher yield).
   - Invert (Long Perp, Short Spot) if funding rate turns deeply negative (< -10.0% annualized).
4. Friction & Slippage Model:
   - Spot taker fee: 7.5 bps
   - Perp taker fee: 4.0 bps
   - Roundtrip entry + exit cost = 23.0 bps.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from backtester import PerformanceAnalytics

def simulate_funding_rate_market(n_days=1000, random_seed=42):
    """Simulates realistic spot crypto prices and 8-hour funding rates."""
    np.random.seed(random_seed)
    
    # Spot price random walk with bull/bear regimes
    daily_vol = 0.035
    trends = np.random.choice([0.0015, -0.001, 0.0002], size=n_days, p=[0.45, 0.35, 0.20])
    daily_returns = np.random.normal(trends, daily_vol)
    spot_price = 40000.0 * np.exp(np.cumsum(daily_returns))
    
    # Funding rates strongly correlate with market momentum & price trends
    # Base funding: 0.01% per 8h (~10.95% APR baseline)
    mom_7d = pd.Series(spot_price).pct_change(7).fillna(0.0).values
    
    # 8h funding rate in basis points (1 bp = 0.01%)
    # Ranges from -15 bps to +50 bps per 8-hour period
    funding_8h_bps = 1.0 + 35.0 * np.tanh(mom_7d * 8.0) + np.random.normal(0, 1.5, size=n_days)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="D")
    return pd.DataFrame({
        "Spot_Price": spot_price,
        "Funding_8h_bps": funding_8h_bps
    }, index=dates)

def run_funding_arbitrage(market_df: pd.DataFrame, entry_apr_threshold=0.10, exit_apr_threshold=0.03):
    # Annualized funding rate: 8h rate * 3 * 365
    market_df["Funding_APR"] = (market_df["Funding_8h_bps"] / 10000.0) * 3 * 365
    
    positions = pd.Series(0.0, index=market_df.index)
    current_pos = 0.0
    
    for t in range(len(market_df)):
        apr = market_df["Funding_APR"].iloc[t]
        
        # Entry: Annualized funding > 10%
        if apr > entry_apr_threshold and current_pos == 0.0:
            current_pos = 1.0  # Enter delta-neutral cash and carry
        # Exit: Funding drops below 3%
        elif apr < exit_apr_threshold and current_pos == 1.0:
            current_pos = 0.0
            
        positions.iloc[t] = current_pos
        
    market_df["Position"] = positions.shift(1).fillna(0.0)
    
    # Daily funding income: 3 periods per day
    daily_funding_yield = 3 * (market_df["Funding_8h_bps"] / 10000.0)
    gross_yield = market_df["Position"] * daily_funding_yield
    
    # Transaction costs on position open/close:
    # 23 bps roundtrip (Spot 7.5 bps + Perp 4 bps each way)
    turnover = market_df["Position"].diff().abs()
    cost = turnover * (23.0 / 10000.0)
    
    net_daily_returns = gross_yield - cost
    return net_daily_returns.iloc[1:]

if __name__ == "__main__":
    market = simulate_funding_rate_market(n_days=1250)
    returns = run_funding_arbitrage(market)
    analytics = PerformanceAnalytics(returns)
    analytics.print_summary("Crypto Delta-Neutral Funding Rate Cash-and-Carry Arbitrage")
