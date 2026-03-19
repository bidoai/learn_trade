# Hedge Fund Trading System — Learning Platform

A complete buy-side trading system built for understanding how a quantitative hedge fund
operates end-to-end. Covers market data ingestion, signal generation, portfolio construction,
order management, risk management, execution, and performance reporting.

**Audience:** This README assumes a background in mathematical finance and sell-side risk
(CCR, XVA). It draws explicit parallels between sell-side and buy-side concepts, explains
the financial intuition behind each component, and covers the math where it matters.

---

## What This System Is (and Is Not)

This system is a learning platform, not a profitable trading system.

It intentionally simplifies or omits several hard problems that dominate real-world performance:

- No transaction costs or market impact in baseline backtests
- No survivorship bias handling in datasets
- No corporate actions (splits, dividends)
- No realistic latency or execution uncertainty
- No borrow constraints for shorting
- No regime detection or adaptive models

As a result, any backtest performance should be assumed to be overstated.

The goal is not to produce alpha, but to understand the full lifecycle of a trading system
and the failure modes that cause most strategies to break in production.

---

## Table of Contents

1. [The Buy-Side Mental Model](#1-the-buy-side-mental-model)
2. [System Architecture](#2-system-architecture)
3. [Market Data Layer](#3-market-data-layer)
4. [Signal Generation (Alpha Research)](#4-signal-generation-alpha-research)
5. [Portfolio Construction](#5-portfolio-construction)
6. [Risk Management](#6-risk-management)
7. [Order Management and Execution](#7-order-management-and-execution)
8. [Performance Attribution](#8-performance-attribution)
9. [Backtesting Pitfalls](#9-backtesting-pitfalls)
10. [Transaction Costs and Slippage](#10-transaction-costs-and-slippage)
11. [Strategy Capacity](#11-strategy-capacity)
12. [Observability and Monitoring](#12-observability-and-monitoring)
13. [The Math](#13-the-math)
14. [What Real Hedge Funds Look Like](#14-what-real-hedge-funds-look-like)
15. [Running the System](#15-running-the-system)
16. [Next Steps](#16-next-steps)
17. [What Would Be Required for Production](#17-what-would-be-required-for-production)

---

## 1. The Buy-Side Mental Model

### Sell-Side vs. Buy-Side Risk: The Core Difference

As a CCR quant you spent your career measuring and pricing risk **you take on from
counterparties** — the exposure that arises when a swap counterparty defaults. The
economic question was: what is the fair cost of this counterparty's credit risk?

The buy side asks a fundamentally different question: **how do I generate returns
that exceed my risk?**

| Concept | Sell-Side (CCR/XVA) | Buy-Side (Hedge Fund) |
|---------|---------------------|----------------------|
| Core question | What does this counterparty risk cost me? | How much alpha can I extract from the market? |
| Risk measure | EPE, PFE, CVA | VaR, Sharpe, max drawdown |
| Time horizon | Deal lifetime (5–30 years) | Minutes to months |
| P&L driver | Spread income, hedging | Alpha minus transaction costs |
| Exposure | Credit exposure to counterparty | Market exposure to factors |
| Wrong-way risk | Counterparty correlated with trade | Strategy correlated with market crash |
| Netting | Netting sets under ISDA | Portfolio-level netting of longs/shorts |
| Collateral | CSA margin calls | Broker margin calls (leverage) |
| Model validation | Exposure model calibration | Backtesting (with its own biases) |

**The most important reframe:** On the sell side, risk is something you measure and
price so you can hedge it away. On the buy side, **risk is what you get paid to take**.
A hedge fund that eliminates all risk makes zero return. The job is to take *the right
risks* in *the right amounts* — risks where you have an edge.

### What Is a Hedge Fund?

A hedge fund is a pool of capital that earns returns by:
1. Identifying **mispricings** (alpha signals)
2. Taking positions to **exploit** those mispricings
3. Managing the **risk** of being wrong
4. Delivering **risk-adjusted returns** to investors (alpha net of fees)

The word "hedge" is historical. Most hedge funds today are more accurately described
as **absolute return** funds — they target positive returns regardless of market direction,
typically by running market-neutral or low-net-exposure books.

### The Alpha Decay Problem

Signals decay. A pricing anomaly that generates 2% annualized alpha today may generate
0.5% in two years as more capital chases it. In practice, this means hedge funds must:
- Constantly research new signals
- Guard their signals as trade secrets
- Diversify across many uncorrelated signals
- Operate in less liquid, less efficient markets where decay is slower

Your CCR analogy: wrong-way risk is a known risk that everyone models now — it wasn't
always. The moment a risk becomes universally modeled and priced, the edge disappears.
Same principle for signals.

---

## 2. System Architecture

### Event-Driven Design

The system is event-driven — components communicate by publishing and subscribing to events
on an internal message bus. This mirrors how real trading infrastructure works.

```
                        ┌─────────────────────────────────────────────────────┐
                        │                   EVENT BUS                          │
                        │         (asyncio fan-out, one Queue per subscriber)  │
                        └───────┬─────────────────────────────────────────────┘
                                │
     ┌──────────────┐           │           ┌──────────────────────────────────┐
     │  Market Data │──publish──┤           │            Subscribers           │
     │  AlpacaWS    │  (bars)   │           │  Strategy Engine  → SignalEvent  │
     └──────────────┘           │           │  Audit Log        → everything   │
                                │           │  Dashboard        → fills/risk   │
     ┌──────────────┐           │           │  Stale Monitor    → MarketData   │
     │  Strategies  │──publish──┤           └──────────────────────────────────┘
     │  Momentum    │  (signals)│
     │  MeanRev     │           │
     │  ML          │           │           ┌──────────────────────────────────┐
     └──────────────┘           │           │     OMS (Order Gatekeeper)        │
                                │           │                                  │
     ┌──────────────┐           │           │  OrderRequest arrives             │
     │  Execution   │──publish──┤           │       │                           │
     │  AlpacaREST  │  (fills)  │           │  [sync] risk_engine.check()       │
     └──────────────┘           │           │       │                           │
                                │           │  approved → AlpacaExecutor       │
     ┌──────────────┐           │           │  blocked  → OrderBlockedEvent    │
     │  Risk Engine │──publish──┘           └──────────────────────────────────┘
     │  (alerts)    │
     └──────────────┘
```

**Why event-driven?** The same reason clearing houses use it: auditability and
decoupling. Every action is a named event with a timestamp. This gives you a
complete audit trail and lets you replay any trading day second-by-second.

### Key Architectural Decision: Risk as Synchronous Gatekeeper

Risk checks are **not** events. The OMS calls `risk_engine.check(order)` as a direct
synchronous function call before any order reaches the broker.

Why? Consider two strategies simultaneously requesting orders. If risk were async
(a subscriber on the bus), both requests could arrive before either fill updates
positions — both could pass the position-size check, doubling your intended exposure.

```
BAD (async risk):                        GOOD (sync gatekeeper):

  OrderA ──▶ Bus ──▶ Risk (async)           OrderA arrives at OMS
  OrderB ──▶ Bus ──▶ Risk (async)                │
                         │                [sync] risk.check(A)  ← positions = 0
          Risk sees                       approved → submit A, positions updated
          position = 0                         │
          for BOTH                        [sync] risk.check(B)  ← positions = 100
          → approves both                 blocked → B exceeds limit
          ← race condition
```

---

## 3. Market Data Layer

### What Comes In

Equity data arrives as OHLCV bars (Open, High, Low, Close, Volume) on a fixed time
interval (1-minute, daily, etc.). In this system we use Alpaca's free paper trading
feed which provides IEX data — real prices, no delay.

```python
@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float    # first trade price in interval
    high: float    # highest trade price
    low: float     # lowest trade price
    close: float   # last trade price (most commonly used for signals)
    volume: int    # number of shares traded
```

The close price is the consensus price at the end of the interval. Volume tells you
conviction — a large move on low volume is less meaningful than the same move on high
volume.

### Why We Use Daily Bars for Learning

In production, HFTs use tick data (every individual trade). For learning:
- Daily bars are sufficient to implement all major strategy types
- Tick data has complex microstructure effects (bid-ask bounce, market impact)
- The signal-to-noise ratio is higher on daily bars

### Stale Data Detection

The `AlpacaWSFeed` monitors for gaps. If a symbol goes quiet for `stale_data_timeout_sec`
(default 60s), a `StaleDataEvent` is published and signal generation halts for that symbol.

**CCR analogy:** This is like market data unavailability in an exposure model. When
you can't observe the market, you conservatively halt rather than extrapolate stale prices.

---

## 4. Signal Generation (Alpha Research)

### What Is Alpha?

Alpha is return that cannot be explained by market exposure (beta). The Capital Asset
Pricing Model (CAPM) tells us:

```
E[R_p] = R_f + β(E[R_m] - R_f) + α

where:
  R_p = portfolio return
  R_f = risk-free rate
  β   = market beta (exposure to systematic market risk)
  E[R_m] - R_f = market risk premium
  α   = excess return not explained by market exposure
```

A pure alpha strategy has β = 0 (market neutral). In practice, most hedge funds have
some small residual beta. The goal is to maximize α while controlling β.

**CCR analogy:** Alpha is like a positive CVA adjustment — it's the extra value your
model generates over the market price. Just as CVA can become negative (you're being
paid less than the risk warrants), alpha can decay to zero.

### Strategy 1: Momentum

**Intuition:** Assets that have recently outperformed tend to continue outperforming
over the next month (Jegadeesh & Titman, 1993). This violates the weak-form Efficient
Market Hypothesis — and yet it persists.

**Why it works (hypotheses):**
- Under-reaction: investors update beliefs slowly, so trends continue
- Feedback loops: rising prices attract capital, driving further rises
- Risk-based: momentum stocks may genuinely have higher systematic risk

**The signal:**
```
signal = (P_t - P_{t-N}) / P_{t-N}    (N-period return)

if signal > threshold → LONG  (positive momentum)
if signal < -threshold → SHORT (negative momentum)
```

**Implementation** (`strategy/momentum.py`):
```python
returns = (newest_close - oldest_close) / oldest_close
direction = clip(returns / (threshold * 5), -1, 1)
```

### Failure Modes: Momentum

- **Crowding:** too many funds running similar signals leads to correlated unwinds — when
  one fund is forced to sell, all momentum portfolios fall together
- **Regime shifts:** momentum underperforms in high-volatility reversals; the strategy
  that worked for 18 months can give back months of gains in days
- **Implicit short volatility exposure:** long momentum is effectively short volatility —
  it collects small gains consistently, then suffers sharp drawdowns during crashes

### Strategy 2: Mean Reversion

**Intuition:** Prices that deviate significantly from their historical mean tend to
revert. This is the contrarian view — it works in range-bound markets where momentum fails.

**The signal (z-score):**
```
μ = mean(P_{t-N}, ..., P_t)
σ = std(P_{t-N}, ..., P_t)
z = (P_t - μ) / σ

if z < -2.0 → LONG  (oversold, expect reversion up)
if z > +2.0 → SHORT (overbought, expect reversion down)
```

**Why z-score?** It normalizes the signal across different symbols and volatility
regimes. A $5 move in AAPL is not the same as a $5 move in a $20 stock. Z-score
expresses the move in units of standard deviation — comparable across assets.

**CCR analogy:** This is exactly the mean-reversion assumption in Hull-White
interest rate models. Rates drift toward a long-run mean θ with speed κ. Here
we're applying the same intuition to equity prices.

### Failure Modes: Mean Reversion

- **Structural breaks:** price does not revert when the move reflects a genuine change —
  earnings miss, accounting fraud, macro shock; you keep buying into a falling knife
- **Volatility clustering:** z-scores expand during stress periods, triggering larger
  positions exactly when the market is most dislocated
- **Hidden trend exposure:** being short momentum (buying laggards, selling leaders)
  means this strategy is implicitly short momentum risk premia

### Strategy 3: ML-Based (Gradient Boosting)

**Intuition:** A supervised model trained on historical features (technical indicators,
volume patterns) predicts next-bar direction. The model learns non-linear relationships
that rule-based strategies miss.

**The pipeline:**
```
Historical bars
      │
      ▼
Feature Engineering
  returns_1d, returns_5d, returns_20d    ← momentum signals at different horizons
  volume_ratio = volume / ma_volume(20)  ← unusual volume flag
  rsi_14                                 ← relative strength index
  bb_position                            ← position within Bollinger bands
  volatility_20d                         ← realized vol regime
      │
      ▼
GradientBoostingClassifier (sklearn)
  Target: 1 if next_close > today_close, else 0
  Training: TimeSeriesSplit (NO random shuffle — see below)
      │
      ▼
Prediction: P(up | features)
  direction = 2 * P(up) - 1   ← maps [0,1] to [-1,+1]
```

**The look-ahead bias trap:** This is the most important concept in ML for finance.
If you train a model using data from 2023 to predict 2022 returns, the model
"knows" the future and will appear to work perfectly in backtest but fail live.

```python
# WRONG: shuffled split leaks future data
X_train, X_test = train_test_split(X, shuffle=True)  ← NEVER DO THIS

# RIGHT: time series split respects causality
tscv = TimeSeriesSplit(n_splits=5)
# Fold 1: train on months 1-2,  test on month 3
# Fold 2: train on months 1-4,  test on month 5
# Fold 3: train on months 1-6,  test on month 7
# etc.
```

The `FeatureBuilder` class enforces this architecturally:
```python
def build(self, df: pd.DataFrame, cutoff_date: datetime) -> pd.DataFrame:
    assert (df.index <= cutoff_date).all(), "Look-ahead bias detected"
    ...
```

**CCR analogy:** Look-ahead bias is exactly the bias that appears when you calibrate
an exposure model using future realized rates — your EPE looks great historically
but the model is useless prospectively.

### Failure Modes: ML Strategy

- **Overfitting to noise:** with a small feature set and short history, the model
  learns patterns that don't generalize — backtests look good, live performance collapses
- **Feature instability across regimes:** features that predict well in low-vol
  trending markets may have zero predictive value during crises
- **Label leakage via improper alignment:** if features are computed with a one-day
  lag but the label is same-day return, you're using tomorrow's data today
- **Non-stationarity:** a model trained on 2018–2022 may fail in 2023 because the
  underlying regime changed; there is no guarantee the past resembles the future

---

## 5. Portfolio Construction

### The Multi-Strategy Problem

Running multiple strategies simultaneously creates a portfolio construction problem:
how much capital should each strategy receive?

**This system uses fixed-weight allocation** (the simplest approach):
```python
strategy_weights = {
    "momentum":       0.40,   # 40% of capital
    "mean_reversion": 0.30,   # 30% of capital
    "ml":             0.30,   # 30% of capital
}
```

**Why multiple strategies?** The Sharpe ratio of an equally-weighted portfolio of
N uncorrelated strategies with individual Sharpe S is:

```
Sharpe_portfolio = S × √N
```

Two uncorrelated strategies each with Sharpe 0.7 produce a combined Sharpe of ~1.0.
This is the core of hedge fund portfolio construction — diversification across
uncorrelated *strategies*, not just securities.

**The correlation problem:** Momentum and mean reversion are naturally anti-correlated
(one is long what the other is short). This is ideal. In practice, strategies become
correlated during market stress — when everyone rushes to the exits, all momentum
strategies crash simultaneously.

### Limitations of Fixed Weights

Fixed weights ignore:
- **Changing volatility across strategies:** a 40% allocation to momentum contributes
  more risk when momentum is volatile than when it is quiet
- **Correlation instability:** strategy correlations shift across regimes; fixed weights
  do not adapt
- **Regime dependence:** a weight that was optimal in trending markets may be wrong
  in mean-reverting markets

In practice, fixed weights mean your realized risk contribution per strategy varies
continuously even though the nominal allocation is constant.

### Suggested Improvement: Volatility Scaling

Scale each strategy by inverse volatility:

```
w_i ∝ 1 / σ_i
```

This stabilizes risk contribution across strategies. When a strategy becomes more
volatile, its allocation shrinks automatically. The result is a portfolio where each
strategy contributes roughly equal risk, not equal capital.

### Suggested Improvement: Correlation-Aware Allocation

Estimate a rolling correlation matrix and reduce allocation to highly correlated
strategies. Two strategies with 0.9 correlation are not providing independent return
streams — allocating to both is largely redundant.

### Suggested Improvement: Capital Allocation as a Control Problem

Portfolio construction should be viewed as:

```
Maximize expected return subject to risk constraints
```

Not: "Assign fixed weights."

In practice, this means solving an optimization at each rebalance:
- Mean-variance (Markowitz): minimize portfolio variance for a given return target
- Risk parity: equalize risk contribution across strategies
- Black-Litterman: blend market equilibrium weights with proprietary views

Each approach has trade-offs. Markowitz is sensitive to estimation error in expected
returns. Risk parity ignores expected return entirely. Black-Litterman requires
explicit return forecasts. Fixed weights avoid all of these problems by ignoring
the data — which is honest but leaves performance on the table.

### Position Sizing

For each signal, position size is determined by:
```
target_value = allocated_capital × |direction| × confidence
quantity = floor(target_value / price)
```

**Real hedge funds use more sophisticated sizing:**

*Kelly Criterion:* Optimal bet size given edge and odds:
```
f* = (bp - q) / b

where:
  b = odds (expected win / expected loss)
  p = probability of winning
  q = 1 - p = probability of losing
```
Full Kelly is often too aggressive. Most quants use half-Kelly or quarter-Kelly.

*Risk parity:* Size positions so each contributes equally to portfolio risk:
```
w_i = (1/σ_i) / Σ(1/σ_j)

where σ_i is the volatility of strategy i
```

*Mean-variance optimization (Markowitz):*
```
min w'Σw  subject to w'μ ≥ target_return, Σw_i = 1, w_i ≥ 0

where:
  Σ = covariance matrix of strategy returns
  μ = vector of expected returns
```

**CCR analogy:** Position sizing is analogous to capital allocation in SA-CCR.
You're distributing a scarce resource (risk budget) across positions to maximize
risk-adjusted return.

---

## 6. Risk Management

### Pre-Trade Risk Checks

Before every order reaches the broker, the risk engine runs synchronous checks:

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Circuit breaker | Halt if daily loss exceeds limit | `circuit_breaker.is_triggered()` |
| Position size | No single position > X% of capital | `position_value > max_position_pct × capital` |
| Concentration | No single symbol > Y% of exposure | `symbol_exposure > max_concentration_pct × capital` |
| Capital check | Sufficient buying power | `position_value > available_capital` |

### The Circuit Breaker

The circuit breaker is the most important risk control. It halts all trading when
the daily loss reaches a threshold (default: 2% of capital).

**Why 2%?** A fund with 20% annual volatility has ~1.25% daily volatility (20%/√252).
A 2% daily stop is approximately 1.6σ — uncommon under normal conditions but reachable
during market stress. It gives strategies enough room to breathe while preventing
catastrophic single-day losses.

**Real funds use risk limits at multiple levels:**
- Position-level: max position per security
- Strategy-level: max drawdown per strategy
- Portfolio-level: max daily/monthly/drawdown loss
- Desk-level: max VaR per desk
- Fund-level: NAV triggers (below certain NAV, stop trading, return capital)

### Value at Risk (VaR)

VaR answers: "What is the maximum loss with 95% confidence over the next day?"

**Historical simulation VaR** (this system):
```
VaR_{95%} = -5th percentile of {r_{t-N}, ..., r_t}

Example:
  Last 20 daily returns: [-1.2%, -0.8%, +0.3%, ..., +1.1%]
  Sorted: [-2.1%, -1.8%, -1.5%, ...]
  VaR_{95%} = 1.5% of portfolio value
```

**CCR analogy:** This is virtually identical to the market risk VaR you'd compute
for IMM (Internal Models Method). The difference: CCR VaR is computed under a
risk-neutral measure for pricing; portfolio VaR is computed under the real-world
measure for risk management.

**VaR limitations** (you already know these from CCR):
- Doesn't capture tail risk beyond the threshold (CVaR/Expected Shortfall does)
- Historical simulation assumes the past represents the future
- During market stress, correlations go to 1 (diversification disappears)
- VaR can be gamed (short vol strategies look great until they don't)

**What real funds measure instead of or alongside VaR:**
- **Expected Shortfall (CVaR):** average loss in the worst 5% of days
- **Maximum drawdown:** peak-to-trough loss over any period
- **Drawdown duration:** how long to recover from a drawdown
- **Sharpe ratio:** return per unit of volatility
- **Calmar ratio:** annualized return / max drawdown
- **Factor exposures:** beta to market, size, value, momentum factors

### Drawdown vs. CCR Exposure

The buy-side risk concept most analogous to CCR exposure is **drawdown path**:

```
CCR (PFE):                              Buy-Side (Drawdown):

Peak Exposure                           Peak NAV
      │                                       │
      │    /\    /\                           │    /\/\
      │   /  \  /  \                          │   /    \
      │  /    \/    \                         │  /      \___
      │ /            \                        │ /
      │/              \                       │/
    ──┼────────────────▶ time               ──┼──────────────▶ time
      │                                       │     ←DD→
      └── EPE = average exposure              └── Max DD = max peak-to-trough

PFE_{95%} = 95th percentile path         DD recovery time = time to new high
```

Maximum drawdown is the primary risk metric investors care about. A fund with a
-40% drawdown requires +67% just to break even — this is the compounding effect
working against you.

---

## 7. Order Management and Execution

### The Order Lifecycle

```
Strategy generates signal
        │
        ▼
PortfolioAllocator sizes the order
  (signal → OrderRequestEvent)
        │
        ▼
OMS receives OrderRequestEvent
        │
  [sync] risk_engine.check()
        │
   approved? ──▶ Order(PENDING_SUBMIT) ──▶ AlpacaExecutor
        │                                       │
   blocked?  ──▶ OrderBlockedEvent          Alpaca API
                                                │
                                         Fill arrives via WebSocket
                                                │
                                         FillEvent ──▶ PositionTracker
                                                    ──▶ PerformanceTracker
                                                    ──▶ Dashboard
```

### Order State Machine

Orders follow a strict state machine (see `oms/state_machine.py`):

```
NEW ──risk_pass──▶ PENDING_SUBMIT ──acknowledged──▶ OPEN ──full_fill──▶ FILLED
 │                                                    │
risk_fail                                          partial_fill ──▶ PARTIAL ──▶ FILLED
 │                                                    │
 ▼                                                 cancel_req ──▶ PENDING_CANCEL ──▶ CANCELLED
BLOCKED                                               │
                                                   timeout ──▶ EXPIRED
```

Invalid transitions raise `InvalidTransitionError` immediately. This makes bugs
loud rather than allowing silent state corruption.

### Execution Quality: Slippage and Market Impact

This system fills orders at the next bar's open price. Real execution is messier:

**Slippage:** The difference between the price you expected and the price you got.
Sources:
- Bid-ask spread (you buy at ask, sell at bid)
- Market impact (your order moves the price)
- Delay between signal and order submission

**Market impact model (Almgren-Chriss):**
```
ΔP = η × σ × (v/V)^α

where:
  η   = market impact coefficient (~1.0)
  σ   = daily volatility of the stock
  v   = order size (shares)
  V   = average daily volume
  α   = impact exponent (~0.5 for square-root law)
```

For a $10M fund trading AAPL ($100B ADV), impact is negligible. For a $10B fund
trading a small-cap ($10M ADV), slippage can easily exceed alpha.

**In practice, this means:** a strategy with 5% gross alpha and 3% annual turnover
costs might net 2%. The same strategy at $1B AUM might have 4% market impact costs,
netting -1%. Capacity — how much capital a strategy can deploy before alpha decays
due to market impact — is a core constraint on hedge fund scalability.

### Execution Risks

Real execution introduces failure modes this system does not model:

- **Partial fills:** large orders may fill across multiple prints at different prices
- **Order rejection:** broker may reject orders due to buying power, risk limits, or
  market conditions
- **Latency between signal and execution:** by the time the order reaches the exchange,
  the price has moved
- **Price drift during execution:** for large orders executed over time, early fills
  move the price against later fills

### Production Reality

Execution is an adversarial environment:
- You are trading against other informed participants
- Your signal may already be priced in by the time you act
- Delays reduce alpha; the half-life of intraday signals can be seconds

In practice, this means backtests should include randomized execution delay and
slippage noise, not clean fills at the bar open. Even small delays (100ms) can
eliminate intraday alpha entirely.

---

## 8. Performance Attribution

### The Sharpe Ratio

The primary metric for evaluating a trading strategy:
```
Sharpe = (μ_p - R_f) / σ_p × √252    (annualized)

where:
  μ_p = mean daily return
  R_f = daily risk-free rate
  σ_p = standard deviation of daily returns
  √252 = annualization factor (252 trading days/year)
```

**Benchmarks:**
- Sharpe < 0.5: poor (most mutual funds)
- Sharpe 0.5–1.0: acceptable
- Sharpe 1.0–2.0: good (institutional quality)
- Sharpe > 2.0: excellent (top quant funds)
- Sharpe > 3.0: suspicious (probably look-ahead bias)

### Information Ratio vs. Sharpe Ratio

If the portfolio has market beta, use the **Information Ratio** instead:
```
IR = α / TE

where:
  α  = excess return vs. benchmark (annualized)
  TE = tracking error = std(R_p - R_benchmark)
```

IR measures the quality of active management. A long-only equity fund with
IR = 0.5 is considered good. A market-neutral quant fund with IR < 1.0 is
probably not worth the fees.

### P&L Attribution

A real fund attributes P&L to understand what's working:

```
Total P&L = Σ (strategy_i P&L)
          = Σ (alpha_i) + Σ (beta_i × market_return) + residual

Per strategy:
  P&L = Σ fills × (exit_price - entry_price)
      = realized P&L (closed positions)
      + unrealized P&L (open positions, mark-to-market)
```

**Factor attribution** goes deeper — decomposing returns by risk factor:
```
R_p = α + β_mkt × R_mkt + β_size × R_size + β_value × R_value
    + β_momentum × R_momentum + ε

where factors are the Fama-French factors (or BARRA risk model)
```

This tells you whether your "alpha" is just disguised beta. A momentum strategy
has natural positive loading on the momentum factor — that's not alpha, that's
a known risk premium. True alpha is the ε residual after removing all factor
exposures.

---

## 9. Backtesting Pitfalls

A backtest is not a simulation of reality — it is an optimistic approximation.

Common biases:

- **Look-ahead bias:** using future information unknowingly — the most common source
  of spectacular backtests that fail immediately in production
- **Survivorship bias:** only including assets that exist today, which excludes
  companies that went bankrupt or were delisted; in practice this inflates returns
- **Selection bias:** choosing strategies or parameters that already worked on the
  data you're testing on
- **Data snooping:** testing too many variations and picking winners — with enough
  trials, random noise will produce a backtest Sharpe > 2.0
- **Ignoring transaction costs and slippage:** all turnover-heavy strategies look
  better without costs; adding realistic costs often halves the Sharpe
- **Ignoring market impact and capacity limits:** a strategy that fills instantly
  at mid-price is not a real strategy at scale

**Rule of thumb:** if a backtest Sharpe > 2.0 without transaction costs, assume
it is overstated until proven otherwise.

Mitigation techniques:

- **Walk-forward validation:** train on a rolling window, test on the next period,
  repeat — never look back at the test periods when adjusting parameters
- **Out-of-sample testing:** hold out the last N years completely; only look at
  them once, after all parameter selection is done
- **Cross-validation with time-series splits:** use `TimeSeriesSplit`, never random
  shuffle on financial time series
- **Adding conservative transaction cost assumptions:** even 5–10 bps per trade
  significantly changes the picture for high-turnover strategies

---

## 10. Transaction Costs and Slippage

Current backtests assume fills at next bar open with no cost. This is unrealistic.

A simple cost model to add:

```
effective_price = price × (1 + sign × cost_bps / 10000)

where:
  sign     = +1 for buy, -1 for sell
  cost_bps = total cost in basis points (spread + slippage + commission)
```

Components of cost_bps:

- **Commission:** 1–5 bps per trade depending on broker and size
- **Bid-ask spread:** 1–10 bps for large-cap equities, much wider for small-cap
- **Slippage:** proportional to volatility and urgency; market orders in thin markets
  can slip 10–50 bps
- **Market impact:** function of order size vs. ADV (see Almgren-Chriss above)

**In practice, this means:** even simple costs significantly reduce Sharpe and can
make turnover-heavy strategies unprofitable. A strategy with daily rebalancing at
10 bps round-trip costs ~25% per year in transaction costs alone. This is the most
important reality check on any backtest.

---

## 11. Strategy Capacity

Every strategy has a maximum capital it can deploy before returns degrade.

Drivers:
- **Market impact:** larger orders move prices more; at some point the trade itself
  is the dominant price movement
- **Liquidity (ADV constraints):** trading too large a fraction of average daily volume
  guarantees adverse slippage
- **Signal decay:** institutional buying pressure in a stock can eliminate the
  mispricing you were trying to exploit

**Rule of thumb:** do not trade more than 1–5% of ADV per day.

In practice, this means a strategy that works at $1M may fail at $100M. The capacity
of a strategy is often more important than its raw Sharpe — a Sharpe 3.0 strategy
capped at $5M is less valuable to a fund than a Sharpe 1.2 strategy that scales to $1B.

Implication for system design: any realistic backtest should impose ADV-based position
limits and simulate market impact, not assume unlimited capacity at mid-price.

---

## 12. Observability and Monitoring

A production trading system must be observable. Without monitoring, failures go
undetected and losses compound silently.

Required components:

- **Structured event logging:** every order, fill, risk breach, and state transition
  logged with timestamp and context — the audit log in this system is the foundation
- **Real-time alerts** for:
  - Risk breaches (circuit breaker triggered, position limit hit)
  - Execution failures (order rejected, fill timeout)
  - Data outages (stale data events, feed disconnection)
- **Health checks** for all components: data feed, OMS, execution, risk engine

The `StaleDataEvent` in this system is an example of the right pattern: a component
detects its own degraded state and broadcasts it, rather than silently producing
bad output.

In production, these alerts route to Slack or PagerDuty. A risk breach at 9:45am
that goes undetected until 3pm can mean hours of unintended exposure.

---

## 13. The Math

### Returns Arithmetic

Daily log return vs. simple return:
```
Simple: r_t = (P_t - P_{t-1}) / P_{t-1}
Log:    r_t = ln(P_t / P_{t-1})

For small moves, log ≈ simple.
Log returns are additive: r_{1→T} = Σ r_t
Simple returns are compoundable: (1+r_{1→T}) = Π (1+r_t)
```

Use log returns for statistics; use simple returns for P&L.

### Realized Volatility

```
σ_realized = std(r_t, ..., r_{t-N}) × √252    (annualized)

Rolling 20-day realized vol is the most common risk measure.
EWMA (λ=0.94) gives more weight to recent returns — used in RiskMetrics:

σ²_t = λ × σ²_{t-1} + (1-λ) × r²_{t-1}
```

### Z-Score (Mean Reversion Signal)

```python
import numpy as np
closes = np.array([bar.close for bar in self.bars])
z = (closes[-1] - closes.mean()) / closes.std()
```

The z-score tells you how many standard deviations the current price is from its
rolling mean. Under the null hypothesis of i.i.d. normal returns, |z| > 2 occurs
about 5% of the time — making it a statistically meaningful signal threshold.

**Note:** Equity prices are NOT i.i.d. normal (they have fat tails, autocorrelation,
and volatility clustering). The z-score is still useful as a relative measure, but
its statistical interpretation requires care.

### Momentum Return

```python
n_bar_return = (closes[-1] - closes[0]) / closes[0]
```

The classic Jegadeesh-Titman signal uses 12-month returns skipping the most recent
month (to avoid short-term reversal contaminating the signal). For daily bars:
```
momentum = return over bars [t-252 : t-21]  (skip last month)
```

### VaR (Historical Simulation)

```python
import numpy as np
returns = np.diff(equity_curve) / equity_curve[:-1]
var_95 = -np.percentile(returns, 5)   # loss at 5th percentile
```

**Parametric VaR** (assumes normal returns):
```
VaR_{95%} = μ - 1.645 × σ    (1-tailed 95%)
VaR_{99%} = μ - 2.326 × σ    (1-tailed 99%)
```

**CVaR (Expected Shortfall):**
```
CVaR_{95%} = -E[R | R < -VaR_{95%}]
           = -(1/0.05) × ∫_{-∞}^{-VaR} r f(r) dr
```

### Sharpe Ratio

```python
daily_returns = np.diff(equity) / equity[:-1]
sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
```

### Maximum Drawdown

```python
peak = np.maximum.accumulate(equity)
drawdown = (peak - equity) / peak
max_drawdown = drawdown.max()   # expressed as fraction of peak
```

### Kelly Criterion

For a binary outcome (win probability p, win amount b, lose amount 1):
```
f* = p - q/b = p - (1-p)/b

Fractional Kelly (recommended): f = f*/k, where k=2 (half-Kelly) or k=4 (quarter-Kelly)
```

For a continuous distribution of returns:
```
f* = μ / σ²    (approximately, for small positions)
```

This is the theoretical optimum — it maximizes long-run wealth growth but
implies extreme volatility. Most funds use 20-25% of Kelly.

---

## 14. What Real Hedge Funds Look Like

### The Technology Stack at a Quant Fund

| Layer | Learning System | Production (~$500M AUM quant fund) |
|-------|----------------|-------------------------------------|
| Data | Alpaca IEX (free) | Bloomberg, Refinitiv, alternative data |
| Storage | SQLite | kdb+ (tick database), S3, ClickHouse |
| Messaging | asyncio Queue | Kafka, ZeroMQ, Aeron |
| OMS | Custom Python | FlexTrade, Fidessa, or in-house C++ |
| Execution | Alpaca REST | FIX protocol to prime broker |
| Risk | Custom Python | Axioma, BARRA, or in-house |
| Languages | Python | Python (research), C++/Java (execution) |
| Latency | Seconds | Microseconds to milliseconds |

### The Quant Fund Workflow

**Research loop** (the quant researcher's job):
```
Idea → Signal construction → Backtest → Paper trading → Live (small size) → Scale up
                                  ↑                          │
                                  └────── iterate ───────────┘
```

**What quant researchers actually do:**
- Read academic papers on market anomalies
- Explore alternative data (satellite imagery, credit card data, NLP on filings)
- Build signal libraries (hundreds of weak signals combined into one strong signal)
- Conduct walk-forward analysis to avoid overfitting
- Write research memos documenting signal decay and capacity

**The PM's job:**
- Allocate capital across strategies
- Manage portfolio-level risk (factor exposures, net beta)
- Make sizing decisions based on risk budget
- Communicate with investors about attribution

### Signal Categories

| Category | Example | Edge Source |
|----------|---------|-------------|
| Technical (price-based) | Momentum, mean reversion | Behavioral finance, risk premia |
| Fundamental | P/E ratio, earnings growth | Information asymmetry |
| Sentiment | Short interest, options flow | Smart money tracking |
| Alternative data | Satellite foot traffic, web scraping | Information advantage |
| Statistical arb | Pairs trading, cointegration | Mean-reversion of spreads |
| Event-driven | M&A, earnings, index rebalancing | Corporate action exploitation |
| Market microstructure | Order flow imbalance, queue position | HFT, exchange access |

### Comparison: Your CCR Background vs. Buy-Side

Things that transfer directly:
- **Risk measurement:** VaR, scenario analysis, stress testing — same tools, different context
- **Stochastic processes:** GBM for prices, OU for mean reversion — you know these from rates
- **Monte Carlo:** backtesting is just Monte Carlo under the historical measure
- **Greeks:** delta-hedging options → factor-hedging equity books (same concept)
- **Correlation:** wrong-way risk intuition directly applies to strategy correlation risk

Things that are new:
- **Alpha vs. beta:** sell-side takes risk away; buy-side takes risk to earn alpha
- **Signal research:** no equivalent on the sell-side — pure buy-side activity
- **Execution quality:** market impact, timing — CCR doesn't care about this
- **Investor relations:** hedge funds report to LPs; sell-side reports to regulators
- **Capacity constraints:** strategies have maximum AUM before alpha decays

---

## 15. Running the System

### Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Get Alpaca paper trading credentials (free)
#    https://alpaca.markets → sign up → Paper Trading → API Keys

# 3. Configure
cp .env.example .env
# Edit .env:
#   ALPACA__API_KEY=your_key
#   ALPACA__SECRET_KEY=your_secret

# 4. Run tests (no credentials needed)
pytest tests/ -v

# 5. Start live paper trading
python scripts/run_live.py
# Dashboard: http://localhost:8000

# 6. Run a backtest
python scripts/run_backtest.py \
  --strategy momentum \
  --symbol AAPL \
  --start 2023-01-01 \
  --end 2023-12-31
```

### Project Structure

```
trading_system/
├── core/               # Events, event bus, domain models
├── config/             # Pydantic settings (reads .env)
├── data/               # Market data feeds
├── strategy/           # Alpha signal generation
├── portfolio/          # Capital allocation, position sizing
├── risk/               # Pre-trade risk, circuit breaker, VaR
├── oms/                # Order lifecycle management
├── execution/          # Broker connectivity (Alpaca + simulator)
├── backtest/           # Backtesting (same code as live)
├── audit/              # Append-only SQLite event log
├── api/                # FastAPI dashboard backend
├── dashboard/          # Real-time HTML dashboard
├── ml/                 # ML strategy pipeline (in progress)
├── tests/              # Test suite (synthetic data, no API)
├── docs/adr/           # Architecture Decision Records
└── scripts/            # Entry points
```

---

## 16. Next Steps

### Immediate (to complete the system)
1. **`ml/` module** — `FeatureBuilder` with cutoff assertion, `TimeSeriesSplit`, `MLStrategy`
2. **Position persistence** — load/save positions on restart via event replay
3. **Wire audit log** — `EventStore` subscribes to all event types in `run_live.py`
4. **Performance tracker** — realized P&L, Sharpe calculation, equity curve persistence

### Phase 2 (deepen understanding)
5. **FIX protocol** — replace Alpaca REST with FIX sessions (industry standard)
6. **Transaction cost model** — add slippage and commission to backtest
7. **Factor exposure** — compute Fama-French factor loadings for the portfolio
8. **Walk-forward analysis** — automate the research → backtest → validate loop

### Phase 3 (advanced topics)
9. **Options strategies** — Black-Scholes, Greeks, delta hedging (your MFin background applies directly)
10. **Statistical arbitrage** — pairs trading via cointegration (Engle-Granger)
11. **Alternative data** — NLP on earnings calls, satellite data pipeline
12. **Portfolio optimization** — Markowitz, risk parity, Black-Litterman

### Suggested Reading

**Core texts:**
- *Active Portfolio Management* — Grinold & Kahn (the bible of quant PMs)
- *Advances in Financial Machine Learning* — Marcos López de Prado (ML pitfalls in finance)
- *Algorithmic Trading* — Ernest Chan (practical, accessible)
- *Inside the Black Box* — Rishi Narang (how quant funds actually work)

**Academic papers:**
- Jegadeesh & Titman (1993) — "Returns to Buying Winners and Selling Losers" (momentum)
- Fama & French (1993) — three-factor model (systematic risk factors)
- Almgren & Chriss (2001) — optimal execution with market impact
- López de Prado (2018) — "The False Strategy Theorem" (why most backtests are wrong)

**From your CCR background, the direct bridges are:**
- *Interest Rate Models* (Brigo & Mercurio) → apply OU process intuition to mean reversion
- *Credit Risk* (McNeil, Frey, Embrechts) → apply copula/correlation intuition to strategy correlation
- Basel III CCR capital → FRTB market risk capital (same framework, different book)

---

## 17. What Would Be Required for Production

To move toward a real trading system:

- **Robust data pipeline:** corporate actions (splits, dividends), survivorship-bias-free
  universe construction, point-in-time data to prevent look-ahead at the data layer
- **Transaction cost and market impact modeling:** commissions, spread, slippage, and
  ADV-based capacity limits built into every backtest by default
- **Portfolio optimization:** risk-aware allocation that adapts to changing volatility
  and correlation, not static weights
- **Persistent state and recovery:** positions, orders, and P&L survive restarts;
  the system can replay the event log to reconstruct state
- **Monitoring and alerting:** structured logs, real-time risk breach alerts, data
  outage detection, health checks for every component
- **Broker abstraction:** not a single-provider dependency on Alpaca; a FIX-based
  execution layer that can route to multiple prime brokers
- **Compliance and audit controls:** immutable audit log, regulatory reporting
  (Form PF, EMIR), pre-trade compliance checks

The current system is a research and learning platform, not an investable system.
The gap between a working backtest and a production-ready fund is largely made up
of the items above — and each one is a non-trivial engineering and quantitative
finance problem in its own right.
