"""
Quant Tool — Quantitative trading research, backtesting, and paper execution for Autobot.
Integrates with quant_trading/ (Alpaca Paper Client, Vectorized Backtester, Strategy Suite).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Locate quant_trading package
_POSSIBLE_QUANT_PATHS = [
    Path(__file__).resolve().parents[2] / "quant_trading",
    Path("c:/Users/User 1/OneDrive/Desktop/projects/django projects/personal projects/autobot/quant_trading"),
]

QUANT_DIR: Optional[Path] = None
for p in _POSSIBLE_QUANT_PATHS:
    if p.exists():
        QUANT_DIR = p
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

try:
    from alpaca_paper_trader import AlpacaPaperClient
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

try:
    from backtester import PerformanceAnalytics
    HAS_BACKTESTER = True
except ImportError:
    HAS_BACKTESTER = False


class Quant:
    """Autobot Quantitative Research & Execution Engine."""

    def __init__(self):
        self._client: Optional[AlpacaPaperClient] = None
        if HAS_ALPACA:
            try:
                self._client = AlpacaPaperClient()
            except Exception:
                self._client = None

    def get_account_status(self) -> Dict[str, Any]:
        """Fetch Alpaca paper trading account status, NAV, and buying power.
        
        Returns:
            Dict with cash, equity, buying_power, and status.
        """
        if not self._client:
            return {"ok": False, "error": "AlpacaPaperClient not initialized or missing credentials."}
        try:
            acc = self._client.get_account()
            return {
                "ok": True,
                "status": acc.get("status"),
                "currency": acc.get("currency", "USD"),
                "equity": float(acc.get("equity", 0.0)),
                "cash": float(acc.get("cash", 0.0)),
                "buying_power": float(acc.get("buying_power", 0.0)),
                "portfolio_value": float(acc.get("portfolio_value", 0.0)),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_positions(self) -> Dict[str, Any]:
        """List currently held open positions in the paper trading account.
        
        Returns:
            List of open positions with symbol, qty, market_value, and unrealized_pl.
        """
        if not self._client:
            return {"ok": False, "error": "AlpacaPaperClient not initialized."}
        try:
            positions = self._client.list_positions()
            summary = []
            for p in positions:
                summary.append({
                    "symbol": p.get("symbol"),
                    "qty": float(p.get("qty", 0.0)),
                    "side": p.get("side"),
                    "market_value": float(p.get("market_value", 0.0)),
                    "cost_basis": float(p.get("cost_basis", 0.0)),
                    "unrealized_pl": float(p.get("unrealized_pl", 0.0)),
                    "unrealized_plpc": float(p.get("unrealized_plpc", 0.0)),
                })
            return {"ok": True, "positions_count": len(summary), "positions": summary}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def submit_paper_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Submit a paper trade order through Alpaca Paper API.
        
        Args:
            symbol: Ticker symbol (e.g. 'AAPL', 'SPY', 'BTCUSD').
            qty: Quantity to trade.
            side: 'buy' or 'sell'.
            order_type: 'market' or 'limit'.
            time_in_force: 'day' or 'gtc'.
            
        Returns:
            Order confirmation details.
        """
        if not self._client:
            return {"ok": False, "error": "AlpacaPaperClient not initialized."}
        try:
            res = self._client.submit_order(
                symbol=symbol.upper(),
                qty=qty,
                side=side.lower(),
                order_type=order_type.lower(),
                time_in_force=time_in_force.lower(),
            )
            return {"ok": True, "order": res}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def emergency_close_all(self) -> Dict[str, Any]:
        """Emergency circuit breaker: liquidate all paper positions immediately."""
        if not self._client:
            return {"ok": False, "error": "AlpacaPaperClient not initialized."}
        try:
            res = self._client.close_all_positions()
            return {"ok": True, "message": "All paper positions closed.", "result": res}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_strategies(self) -> Dict[str, Any]:
        """List all available quantitative strategies in quant_trading/strategies/."""
        if not QUANT_DIR:
            return {"ok": False, "error": "quant_trading directory not found."}
        strat_dir = QUANT_DIR / "strategies"
        if not strat_dir.exists():
            return {"ok": True, "strategies": []}
        strats = [f.stem for f in strat_dir.glob("*.py") if not f.name.startswith("__")]
        return {"ok": True, "strategies": strats}
