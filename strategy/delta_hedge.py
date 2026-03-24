"""
Delta-neutral hedging strategy.

Holds a fixed options position and dynamically hedges the delta by trading
the underlying. On each bar:
  1. Compute current delta from Black-Scholes (using realized vol)
  2. Compute target hedge = -delta × contracts × multiplier
  3. Generate a SignalEvent to close the gap between current and target
  4. Publish GreeksEvent with all five Greeks
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional

import structlog

import pricing.black_scholes as bs
from config.settings import StrategySettings
from core.events import GreeksEvent, MarketDataEvent, SignalEvent
from core.models import OptionContract, OptionType
from strategy.base import Strategy

logger = structlog.get_logger(__name__)


class DeltaHedgeStrategy(Strategy):
    """
    Delta-neutral hedging strategy.

    Args:
        option: The OptionContract to hedge.
        settings: StrategySettings.
        risk_free_rate: Annual risk-free rate (default 0.05 = 5%).
        vol_lookback: Days of history to use for realized vol (default 20).
        event_bus: Optional EventBus to publish GreeksEvent on each bar.
    """
    strategy_id = "delta_hedge"

    def __init__(
        self,
        option: OptionContract,
        settings: StrategySettings,
        risk_free_rate: float = 0.05,
        vol_lookback: int = 20,
        event_bus=None,
    ) -> None:
        self.lookback_window = vol_lookback
        super().__init__()
        self._option = option
        self._r = risk_free_rate
        self._vol_lookback = vol_lookback
        self._event_bus = event_bus
        self._current_hedge: float = 0.0
        self._log = logger.bind(strategy=self.strategy_id)

    # ------------------------------------------------------------------
    # Strategy implementation
    # ------------------------------------------------------------------

    def generate_signal(self) -> Optional[SignalEvent]:
        bars = list(self.bars)
        S = bars[-1].close
        symbol = bars[-1].symbol
        today: date = bars[-1].timestamp.date() if hasattr(bars[-1].timestamp, "date") else date.today()

        # Time to expiry in years
        days_to_expiry = (self._option.expiry - today).days
        T = max(days_to_expiry / 252, 0.0)

        # Realized vol from log returns of the lookback window
        closes = [b.close for b in bars]
        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        if len(log_returns) < 2:
            return None
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((x - mean_r) ** 2 for x in log_returns) / (len(log_returns) - 1)
        sigma = math.sqrt(variance * 252)  # annualize

        if sigma <= 0:
            return None

        K = self._option.strike
        option_type_str = self._option.option_type.value  # "call" or "put"

        # Compute all Greeks
        try:
            current_delta = bs.delta(S, K, self._r, T, sigma, option_type_str)
            current_gamma = bs.gamma(S, K, self._r, T, sigma)
            current_vega = bs.vega(S, K, self._r, T, sigma)
            current_theta = bs.theta(S, K, self._r, T, sigma, option_type_str)
            current_rho = bs.rho(S, K, self._r, T, sigma, option_type_str)
        except ValueError:
            return None

        # Publish GreeksEvent if we have an event bus
        if self._event_bus is not None:
            import asyncio
            greeks_event = GreeksEvent(
                symbol=symbol,
                delta=current_delta,
                gamma=current_gamma,
                vega=current_vega,
                theta=current_theta,
                rho=current_rho,
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._event_bus.publish(greeks_event))
                else:
                    loop.run_until_complete(self._event_bus.publish(greeks_event))
            except RuntimeError:
                pass

        # Target hedge: negative of option delta × position size
        contracts = self._option.contracts
        multiplier = self._option.multiplier
        target_shares = -current_delta * contracts * multiplier

        gap = target_shares - self._current_hedge
        max_position = contracts * multiplier

        if max_position == 0 or gap == 0:
            return None

        # Direction: +1 = buy more underlying, -1 = sell underlying
        direction = 1.0 if gap > 0 else -1.0
        confidence = min(1.0, abs(gap) / max_position)

        # Update tracked hedge immediately (optimistic — real system would update on fill)
        self._current_hedge = target_shares

        return SignalEvent(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            reason=f"delta_hedge: delta={current_delta:.4f}, gap={gap:.1f} shares",
        )

    def _reset_state(self) -> None:
        self._current_hedge = 0.0
