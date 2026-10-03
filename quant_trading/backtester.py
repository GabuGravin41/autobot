"""
Autobot Quant Research & Execution Engine: Vectorized Backtester & KPI Evaluator
Computes institutional-grade risk metrics: Sharpe, Sortino, Calmar, Max Drawdown, CVaR, and Kelly Fraction.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class PerformanceAnalytics:
    """Calculates risk and return performance metrics for quantitative strategies."""

    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.04):
        self.returns = returns.dropna()
        self.rf_daily = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0

    def compute_all_kpis(self) -> Dict[str, Any]:
        if len(self.returns) == 0:
            return {}

        n_days = len(self.returns)
        cum_ret = (1.0 + self.returns).cumprod()
        total_return = cum_ret.iloc[-1] - 1.0
        cagr = (cum_ret.iloc[-1] ** (252.0 / n_days)) - 1.0 if n_days > 0 else 0.0

        # Volatility
        vol_ann = self.returns.std() * np.sqrt(252.0)

        # Excess Returns
        excess_ret = self.returns - self.rf_daily

        # Sharpe Ratio
        sharpe = (excess_ret.mean() / (self.returns.std() + 1e-8)) * np.sqrt(252.0)

        # Sortino Ratio (Downside deviation only)
        downside_returns = self.returns[self.returns < 0.0]
        downside_std = downside_returns.std() * np.sqrt(252.0)
        sortino = (excess_ret.mean() * 252.0) / (downside_std + 1e-8) if downside_std > 0 else np.nan

        # Drawdowns
        running_max = cum_ret.cummax()
        drawdowns = (cum_ret - running_max) / running_max
        max_dd = drawdowns.min()

        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

        # Win Rate & Profit Factor
        winning_days = self.returns[self.returns > 0.0]
        losing_days = self.returns[self.returns < 0.0]
        win_rate = len(winning_days) / n_days if n_days > 0 else 0.0
        gross_profit = winning_days.sum()
        gross_loss = abs(losing_days.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

        # Value at Risk (VaR 95%) & Conditional VaR (Expected Shortfall)
        var_95 = np.percentile(self.returns, 5.0)
        cvar_95 = self.returns[self.returns <= var_95].mean()

        # Empirical Kelly Criterion Fraction: f* = (p * b - q) / b
        avg_win = winning_days.mean() if len(winning_days) > 0 else 0.0
        avg_loss = abs(losing_days.mean()) if len(losing_days) > 0 else 1.0
        b_ratio = avg_win / (avg_loss + 1e-8)
        p = win_rate
        q = 1.0 - p
        kelly_fraction = (p * b_ratio - q) / (b_ratio + 1e-8) if b_ratio > 0 else 0.0

        return {
            "Total_Return": float(total_return),
            "CAGR": float(cagr),
            "Annualized_Vol": float(vol_ann),
            "Sharpe_Ratio": float(sharpe),
            "Sortino_Ratio": float(sortino),
            "Max_Drawdown": float(max_dd),
            "Calmar_Ratio": float(calmar),
            "Win_Rate": float(win_rate),
            "Profit_Factor": float(profit_factor),
            "Daily_VaR_95": float(var_95),
            "Daily_CVaR_95": float(cvar_95),
            "Optimal_Kelly_Fraction": float(np.clip(kelly_fraction, 0.0, 1.0)),
        }

    def print_summary(self, strategy_name: str = "Quantitative Strategy"):
        kpis = self.compute_all_kpis()
        print(f"\n=======================================================")
        print(f"  PERFORMANCE SUMMARY: {strategy_name}")
        print(f"=======================================================")
        print(f"  Total Return:          {kpis['Total_Return']:>10.2%}")
        print(f"  CAGR:                  {kpis['CAGR']:>10.2%}")
        print(f"  Annualized Volatility: {kpis['Annualized_Vol']:>10.2%}")
        print(f"  Sharpe Ratio (Rf=4%):  {kpis['Sharpe_Ratio']:>10.2f}")
        print(f"  Sortino Ratio:         {kpis['Sortino_Ratio']:>10.2f}")
        print(f"  Max Drawdown:          {kpis['Max_Drawdown']:>10.2%}")
        print(f"  Calmar Ratio:          {kpis['Calmar_Ratio']:>10.2f}")
        print(f"  Daily Win Rate:        {kpis['Win_Rate']:>10.2%}")
        print(f"  Profit Factor:         {kpis['Profit_Factor']:>10.2f}")
        print(f"  Value at Risk (95%):   {kpis['Daily_VaR_95']:>10.2%}")
        print(f"  Expected Shortfall:    {kpis['Daily_CVaR_95']:>10.2%}")
        print(f"  Half-Kelly Leverage:   {kpis['Optimal_Kelly_Fraction']/2.0:>10.2f}x")
        print(f"=======================================================\n")
