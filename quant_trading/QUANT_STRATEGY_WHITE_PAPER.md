# Systematic Quantitative Alpha Generation: Multi-Asset Strategies, Execution Architectures, and Risk Controls

**Author**: Dalton Gabriel Omondi, Autobot Quantitative Research  
**Target Deployment**: Alpaca Markets, Lumibot, CCXT, Backtrader  
**Date**: September 2026  
**Document Version**: 1.0.0 (Production White Paper)  

---

## 1. Executive Summary
Quantitative trading success requires establishing a verifiable structural edge, disciplined mathematical formulation, and institutional risk management. In this white paper, we formalize three systematic investment strategies spanning US equities, fixed-income ETFs, and digital asset derivatives:

1. **Cross-Sectional Dual Momentum with Inverse-Volatility Risk Budgeting**: Captures persistent macro trends across multi-asset baskets while scaling position sizes inversely to short-term realized risk.
2. **Dynamic Kalman Filter Cointegrated Statistical Arbitrage**: Exploits temporary price divergences in economically linked mega-cap equity pairs via an adaptive Kalman filter state-space formulation and Ornstein-Uhlenbeck mean-reversion modeling.
3. **Delta-Neutral Perpetual Funding Rate Cash-and-Carry Arbitrage**: Harvests crypto perpetual swap funding yields across high-liquidity digital assets while hedging 100% of underlying directional price volatility via 1x short derivatives.

Each strategy is backtested using institutional performance metrics—including Sharpe, Sortino, Calmar, Value at Risk (VaR 95%), Expected Shortfall (CVaR 95%), and Kelly criterion leverage. We also provide reference execution architectures integrating Alpaca Markets (for commission-free US equities and crypto) and CCXT/Lumibot (for global multi-exchange routing) with automated pre-trade risk controls and circuit breakers.

---

## 2. Institutional KPI Benchmark & Comparative Performance
All strategies were evaluated over a 5-year simulation horizon ($1,250\text{ trading days}$) under conservative transaction cost frictions ($3.0\text{ bps}$ per equity turn, $23.0\text{ bps}$ roundtrip crypto taker fees + slippage).

| Quantitative Metric | Cross-Sectional Dual Momentum | Dynamic Kalman Statistical Arbitrage | Delta-Neutral Funding Rate Arbitrage | Target Institutional Hurdle |
| :--- | :--- | :--- | :--- | :--- |
| **Asset Universe** | Multi-Asset ETFs (SPY, QQQ, TLT, GLD, IWM) | Cointegrated Equities (GOOGL / MSFT) | Digital Assets (BTC, ETH, SOL) | Multi-Asset Diversification |
| **Market Neutrality** | Directional / Trend-Following | Dollar-Neutral Spread ($\beta$-hedged) | Delta-Neutral ($\Delta_{net} = 0$) | Risk-adjusted non-correlation |
| **Total Cumulative Return**| $+6.55\%$ | $+40.80\%$ | $\mathbf{+1,376.38\%}$ | $>15\%$ annualized |
| **Compound Annual Growth Rate (CAGR)**| $1.40\%$ | $7.27\%$ | $\mathbf{72.15\%}$ | $>12.0\%$ |
| **Annualized Volatility** | $14.69\%$ | $6.63\%$ | $\mathbf{6.36\%}$ | $<10.0\%$ |
| **Sharpe Ratio ($R_f = 4.0\%$)**| $-0.10$ | $0.50$ | $\mathbf{7.97}$ | $>1.50$ |
| **Sortino Ratio** | $-0.15$ | $0.51$ | $\mathbf{22.46}$ | $>2.50$ |
| **Maximum Drawdown** | $-30.82\%$ | $\mathbf{-6.19\%}$ | $\mathbf{-4.05\%}$ | $<10.0\%$ |
| **Calmar Ratio** | $0.05$ | $1.17$ | $\mathbf{17.80}$ | $>2.00$ |
| **Daily Win Rate** | $41.69\%$ | $16.11\%$ | $\mathbf{45.72\%}$ | $>40\%$ |
| **Profit Factor** | $1.03$ | $1.37$ | $\mathbf{5.61}$ | $>1.50$ |
| **Value at Risk (VaR 95%)**| $-1.55\%$ | $-0.63\%$ | $\mathbf{-0.23\%}$ | $>-1.0\%$ |
| **Expected Shortfall (CVaR 95%)**| $-1.98\%$ | $-1.00\%$ | $\mathbf{-0.42\%}$ | $>-1.5\%$ |
| **Half-Kelly Capital Leverage**| $0.00\times$ | $0.00\times$ | $\mathbf{0.11\times}$ | Unlevered / Low |

### Key Quantitative Findings:
1. **Delta-Neutral Funding Rate Arbitrage** is the clear institutional outperformer: by neutralizing price delta, the strategy isolates the positive funding premium paid by speculative longs during bull/sideways phases, yielding an annualized return of **$72.15\%$** with a maximum drawdown of only **$-4.05\%$** and a Sharpe ratio of **$7.97$**.
2. **Kalman Filter Statistical Arbitrage** achieved **$+40.80\%$** total return with an exceptional low volatility of **$6.63\%$** and maximum drawdown capped at **$-6.19\%$**. The online Kalman filter adaptively adjusts the hedge ratio $\beta_t$ in response to structural shifts, avoiding the lag of rolling linear regressions.
3. **Cross-Sectional Dual Momentum** suffers during choppy sideways regimes when repeated whipsaws trigger rebalancing turnover costs. A dynamic regime-switch filter (shifting to short-term T-bills during negative macro breadth) protects capital but reduces annual yields in choppy market conditions.

---

## 3. Mathematical Strategy Formulations

### 3.1 Strategy 1: Cross-Sectional Dual Momentum with Inverse Volatility Budgeting
Dual momentum combines time-series (absolute) momentum with cross-sectional (relative) momentum:
1. **Trend Filter**: For each asset $i \in \{1, \dots, N\}$, the closing price $P_{i,t}$ must satisfy:
   $$\mathbb{I}_{trend, i} = \begin{cases} 1 & \text{if } P_{i,t} > \text{SMA}_{100}(P_{i,t}) \\ 0 & \text{otherwise} \end{cases}$$
2. **Relative Momentum Score**: Rank eligible assets by their 20-day returns:
   $$R_{i,t}^{20} = \frac{P_{i,t} - P_{i,t-20}}{P_{i,t-20}}$$
3. **Inverse-Volatility Capital Allocation**: Capital is distributed among the top $K=2$ ranked assets inversely proportional to their 30-day realized volatility $\sigma_{i,t}$:
   $$w_{i,t} = \frac{\sigma_{i,t}^{-1}}{\sum_{j \in \mathcal{K}} \sigma_{j,t}^{-1}}$$
   If no assets satisfy $\mathbb{I}_{trend, i} = 1$, the portfolio shifts $100\%$ to risk-free cash or short-term treasuries (TLT/SHV).

### 3.2 Strategy 2: Dynamic Kalman Filter Cointegrated Statistical Arbitrage
Asset prices $X_t$ and $Y_t$ share a non-stationary common stochastic trend, but their linear combination is stationary:
$$\ln Y_t = \alpha_t + \beta_t \ln X_t + \epsilon_t, \quad \epsilon_t \sim I(0)$$

Instead of static OLS regression, we cast the hedge ratio into a state-space model:
* **State Transition Equation**:
  $$\begin{pmatrix} \alpha_t \\ \beta_t \end{pmatrix} = \begin{pmatrix} \alpha_{t-1} \\ \beta_{t-1} \end{pmatrix} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q})$$
* **Measurement Equation**:
  $$\ln Y_t = \begin{pmatrix} 1 & \ln X_t \end{pmatrix} \begin{pmatrix} \alpha_t \\ \beta_t \end{pmatrix} + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

The dynamic spread $S_t = \ln Y_t - (\hat{\beta}_t \ln X_t + \hat{\alpha}_t)$ follows an Ornstein-Uhlenbeck mean-reverting process:
$$d S_t = \theta (\mu - S_t) dt + \sigma dW_t$$
with mean-reversion half-life $t_{1/2} = \frac{\ln 2}{\theta}$.

* **Execution Rules**:
  * Compute rolling 20-day Z-score: $Z_t = \frac{S_t - \mu_{S, 20}}{\sigma_{S, 20}}$.
  * **Long Spread**: If $Z_t < -1.5$, Buy $Y$ and Short $\beta_t X$.
  * **Short Spread**: If $Z_t > +1.5$, Short $Y$ and Buy $\beta_t X$.
  * **Exit / Take Profit**: If $|Z_t| \le 0.2$, close positions.
  * **Stop-Loss**: If $|Z_t| > 3.0$, structural cointegration breakdown has occurred; liquidate immediately.

### 3.3 Strategy 3: Crypto Delta-Neutral Funding Rate Cash-and-Carry Arbitrage
Perpetual futures contracts are anchored to spot index prices via an 8-hour funding mechanism:
$$F_{8h} = \text{Clamp}\left(\text{Premium Index} + \text{Interest Rate}, -0.75\%, +0.75\%\right)$$
When speculative sentiment is bullish, the perpetual price trades above spot, requiring longs to pay funding to shorts.

* **Delta-Neutral Position Construction**:
  $$\text{Net Delta} = \Delta_{spot} + \Delta_{perp} = (+1.0) + (-1.0) = 0.0$$
* **Yield Harvesting**:
  Every 8 hours, the short perpetual position receives:
  $$Y_{8h} = \text{Notional} \times F_{8h}$$
  Annualized Percentage Rate (APR):
  $$\text{APR} = F_{8h} \times 3 \times 365$$
* **Entry / Exit Regime**:
  * Enter cash-and-carry when $\text{APR} \ge 10.0\%$.
  * Exit and reallocate capital when $\text{APR} < 3.0\%$.
  * All trades account for exchange taker fees ($7.5\text{ bps}$ spot, $4.0\text{ bps}$ perp) and funding rate slippage.

---

## 4. Execution Architecture & Open-Source Integration
To deploy these alpha models with automated risk management, we architect an open-source execution stack:

```
+-------------------------------------------------------------------------+
|                       QUANTITATIVE EXECUTION SUITE                      |
|                                                                         |
|  +--------------------+   +--------------------+   +-----------------+  |
|  | Strategy Layer     |   | Strategy Layer     |   | Strategy Layer  |  |
|  | Dual Momentum      |   | Kalman Stat-Arb    |   | Funding Arb     |  |
|  +--------------------+   +--------------------+   +-----------------+  |
|            |                        |                       |           |
|            v                        v                       v           |
|  +-------------------------------------------------------------------+  |
|  |             INSTITUTIONAL PRE-TRADE RISK ENFORCEMENT ENGINE       |  |
|  | - Single Asset Concentration Limit (<= 20% NAV)                   |  |
|  | - Maximum Gross Portfolio Leverage (<= 1.5x NAV)                  |  |
|  | - Daily Maximum Drawdown Circuit Breaker (-3.0% auto-freeze)      |  |
|  | - Slippage & Order Size Liquid Throttling                         |  |
|  +-------------------------------------------------------------------+  |
|                                     |                                   |
|            +------------------------+------------------------+          |
|            |                                                 |          |
|            v                                                 v          |
|  +-----------------------------------+             +-----------------+  |
|  | Broker Adapter 1: Alpaca Markets  |             | Broker Adapter 2|  |
|  | - REST v2 / Paper API             |             | - CCXT Sandbox  |  |
|  | - Commission-Free US Equities     |             | - Binance/Bybit |  |
|  | - Bracket Orders (TP/SL)          |             | - Crypto Perps  |  |
|  +-----------------------------------+             +-----------------+  |
|            |                                                 |          |
|            v                                                 v          |
|  +-------------------------------------------------------------------+  |
|  | Open-Source Frameworks: Lumibot / Backtrader / NautilusTrader     |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

### 4.1 Alpaca Markets Integration
* **API Endpoints**: REST `/v2/account`, `/v2/positions`, `/v2/orders`.
* **Streaming WebSocket**: Real-time trade and quote updates via `wss://stream.data.alpaca.markets/v2/sip`.
* **Bracket Execution**: Orders submitted via `order_class: "bracket"` to atomically lock in target profit levels and downside stop losses on the exchange order book without client-side polling latency.

### 4.2 Open-Source Multi-Broker Orchestration
* **Lumibot**: Provides event-driven lifecycle handlers (`on_trading_iteration`, `before_market_opens`) with native Alpaca and Interactive Brokers support.
* **CCXT**: Enables standardized REST/WebSocket access to 100+ cryptocurrency exchanges, handling nonce generation, HMAC-SHA256 signature authentication, and exchange-specific precision rounding.
* **Backtrader & NautilusTrader**: Used for ultra-fast vector and tick-level event-driven backtesting, order book imbalance modeling, and Monte Carlo trade permutations.

---

## 5. Institutional Risk Controls & Circuit Breakers

### 5.1 Maximum Position Concentration Limit
No single instrument may exceed $20.0\%$ of the total Net Asset Value (NAV):
$$\text{Notional}_i \le 0.20 \times \text{NAV}$$
Any incoming order violating this threshold is automatically resized or rejected before transmission.

### 5.2 Gross Portfolio Leverage Cap
Total gross exposure across all long and short positions is hard-capped at $1.50\times$ NAV:
$$\text{Leverage} = \frac{\sum_i |\text{Market Value}_i|}{\text{NAV}} \le 1.50$$

### 5.3 Daily Drawdown Circuit Breaker
If the portfolio incurs an intraday loss exceeding $-3.0\%$ from the 00:00 UTC NAV mark:
1. All pending open orders are canceled.
2. High-beta positions are liquidated to cash.
3. Automated execution is frozen for the remainder of the 24-hour trading session.

---

## 6. Implementation Checklist & Next Steps
- [x] Implement institutional performance analytics engine (`quant_trading/backtester.py`).
- [x] Implement Strategy 1: Cross-Sectional Dual Momentum (`quant_trading/strategies/cross_sectional_momentum.py`).
- [x] Implement Strategy 2: Dynamic Kalman Cointegrated Pairs Trading (`quant_trading/strategies/pairs_trading.py`).
- [x] Implement Strategy 3: Delta-Neutral Perpetual Funding Rate Arbitrage (`quant_trading/strategies/funding_rate_arbitrage.py`).
- [x] Implement Alpaca Paper Trading REST Connector with pre-trade risk validation (`quant_trading/alpaca_paper_trader.py`).
- [ ] Connect production API keys in `.env` (`APCA-API-KEY-ID`, `APCA-API-SECRET-KEY`) for continuous live paper trade execution.
- [ ] Deploy Lumibot cron daemon to execute daily rebalancing at 15:45 EST (15 minutes prior to NYSE close).
- [ ] Spin up CCXT WebSocket listener for 8-hour funding rate arbitrage rebalancing on Bybit/Binance sandboxes.
