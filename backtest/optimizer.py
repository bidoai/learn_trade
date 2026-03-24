"""
Strategy parameter optimizer with walk-forward validation.

Prevents overfitting by evaluating optimized parameters on unseen out-of-sample
data. Uses Optuna for Bayesian optimization (faster convergence than grid search).

Walk-forward design (expanding window):
    Full range: [P0 | P1 | P2 | P3 | P4]  (n_splits=4 → 5 segments)
    Fold 0: train=P0,          test=P1
    Fold 1: train=P0+P1,       test=P2
    Fold 2: train=P0+P1+P2,    test=P3
    Fold 3: train=P0+P1+P2+P3, test=P4

Each fold optimizes params on in-sample, then scores on out-of-sample.
Reporting both is deliberate: in-sample Sharpe >> OOS Sharpe signals overfitting.

Usage:
    from backtest.optimizer import StrategyOptimizer
    from strategy.momentum import MomentumStrategy
    from config.settings import StrategySettings

    def make_momentum(params):
        s = StrategySettings(
            momentum_lookback=params["lookback"],
            momentum_threshold=params["threshold"],
        )
        return MomentumStrategy(s)

    param_space = {
        "lookback":  {"type": "int",   "low": 5,     "high": 50},
        "threshold": {"type": "float", "low": 0.005, "high": 0.05},
    }

    result = await StrategyOptimizer(
        strategy_factory=make_momentum,
        param_space=param_space,
        symbol="AAPL",
        start=date(2020, 1, 1),
        end=date(2023, 12, 31),
        settings=settings,
        n_splits=4,
        n_trials=50,
    ).run()

    print(result)
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import structlog

from config.settings import Settings
from core.events import MarketDataEvent
from core.models import Bar
from data.historical import HistoricalLoader
from strategy.base import Strategy

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    best_params: Dict[str, Any]
    in_sample_sharpe: float
    out_of_sample_sharpe: Optional[float]
    out_of_sample_return_pct: Optional[float]

    def __str__(self) -> str:
        oos = f"{self.out_of_sample_sharpe:.2f}" if self.out_of_sample_sharpe is not None else "N/A"
        ret = f"{self.out_of_sample_return_pct:+.1f}%" if self.out_of_sample_return_pct is not None else "N/A"
        return (
            f"  Fold {self.fold}: train={self.train_start}→{self.train_end} "
            f"test={self.test_start}→{self.test_end}\n"
            f"    Best params: {self.best_params}\n"
            f"    In-sample Sharpe: {self.in_sample_sharpe:.2f}  "
            f"Out-of-sample Sharpe: {oos}  Return: {ret}"
        )


@dataclass
class OptimizationResult:
    strategy_id: str
    symbol: str
    start: date
    end: date
    n_splits: int
    n_trials: int
    folds: List[FoldResult] = field(default_factory=list)

    @property
    def mean_oos_sharpe(self) -> Optional[float]:
        vals = [f.out_of_sample_sharpe for f in self.folds if f.out_of_sample_sharpe is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def best_params(self) -> Dict[str, Any]:
        """Params from the fold with the highest OOS Sharpe."""
        valid = [f for f in self.folds if f.out_of_sample_sharpe is not None]
        if not valid:
            return {}
        return max(valid, key=lambda f: f.out_of_sample_sharpe).best_params

    def __str__(self) -> str:
        lines = [
            f"OptimizationResult: {self.strategy_id} on {self.symbol} "
            f"({self.start} → {self.end})",
            f"  Walk-forward folds: {self.n_splits}  Optuna trials/fold: {self.n_trials}",
        ]
        for fold in self.folds:
            lines.append(str(fold))
        mean = self.mean_oos_sharpe
        lines.append(
            f"\n  Mean OOS Sharpe: {mean:.2f}" if mean is not None else "\n  Mean OOS Sharpe: N/A"
        )
        lines.append(f"  Recommended params: {self.best_params}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class StrategyOptimizer:
    """
    Walk-forward optimizer using Bayesian search (Optuna) per fold.

    Args:
        strategy_factory: Callable that takes a param dict and returns a Strategy.
        param_space: Dict mapping param names to {"type": "int"|"float", "low": x, "high": y}.
        symbol: Ticker to optimize on.
        start / end: Full date range (will be split into n_splits+1 segments).
        settings: System settings (used for capital, risk params).
        n_splits: Number of walk-forward folds (default 4).
        n_trials: Optuna trials per fold (default 50).
    """

    def __init__(
        self,
        strategy_factory: Callable[[Dict[str, Any]], Strategy],
        param_space: Dict[str, Dict],
        symbol: str,
        start: date,
        end: date,
        settings: Settings,
        n_splits: int = 4,
        n_trials: int = 50,
    ) -> None:
        self.strategy_factory = strategy_factory
        self.param_space = param_space
        self.symbol = symbol
        self.start = start
        self.end = end
        self.settings = settings
        self.n_splits = n_splits
        self.n_trials = n_trials

    async def run(self) -> OptimizationResult:
        """Load data once, then run walk-forward optimization."""
        logger.info(
            "optimizer.starting",
            symbol=self.symbol,
            start=str(self.start),
            end=str(self.end),
            n_splits=self.n_splits,
            n_trials=self.n_trials,
        )

        loader = HistoricalLoader(
            api_key=self.settings.alpaca.api_key,
            secret_key=self.settings.alpaca.secret_key,
        )
        all_bars = loader.load(self.symbol, self.start, self.end)

        if len(all_bars) < (self.n_splits + 1) * 30:
            raise ValueError(
                f"Not enough bars ({len(all_bars)}) for {self.n_splits} splits. "
                "Need at least 30 bars per segment."
            )

        segments = _split_bars(all_bars, self.n_splits + 1)

        result = OptimizationResult(
            strategy_id=self.strategy_factory({k: v["low"] for k, v in self.param_space.items()}).strategy_id,
            symbol=self.symbol,
            start=self.start,
            end=self.end,
            n_splits=self.n_splits,
            n_trials=self.n_trials,
        )

        for fold_idx in range(self.n_splits):
            # Expanding window: train on segments 0..fold_idx, test on fold_idx+1
            train_bars = [b for seg in segments[: fold_idx + 1] for b in seg]
            test_bars = segments[fold_idx + 1]

            logger.info(
                "optimizer.fold",
                fold=fold_idx,
                train_bars=len(train_bars),
                test_bars=len(test_bars),
            )

            best_params, is_sharpe = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda tb=train_bars: self._optimize_fold(tb),
            )

            oos_sharpe, oos_return = _run_backtest_sync(
                self.strategy_factory(best_params), test_bars, self.settings
            )

            fold_result = FoldResult(
                fold=fold_idx,
                train_start=train_bars[0].timestamp.date() if hasattr(train_bars[0].timestamp, "date") else train_bars[0].timestamp,
                train_end=train_bars[-1].timestamp.date() if hasattr(train_bars[-1].timestamp, "date") else train_bars[-1].timestamp,
                test_start=test_bars[0].timestamp.date() if hasattr(test_bars[0].timestamp, "date") else test_bars[0].timestamp,
                test_end=test_bars[-1].timestamp.date() if hasattr(test_bars[-1].timestamp, "date") else test_bars[-1].timestamp,
                best_params=best_params,
                in_sample_sharpe=is_sharpe,
                out_of_sample_sharpe=oos_sharpe,
                out_of_sample_return_pct=oos_return,
            )
            result.folds.append(fold_result)
            logger.info(
                "optimizer.fold_complete",
                fold=fold_idx,
                best_params=best_params,
                in_sample_sharpe=round(is_sharpe, 3),
                oos_sharpe=round(oos_sharpe, 3) if oos_sharpe else None,
            )

        logger.info(
            "optimizer.complete",
            mean_oos_sharpe=round(result.mean_oos_sharpe, 3) if result.mean_oos_sharpe else None,
            best_params=result.best_params,
        )
        return result

    def _optimize_fold(self, train_bars: list[Bar]) -> tuple[Dict[str, Any], float]:
        """Synchronous Optuna study for one fold. Runs in a thread executor."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        strategy_factory = self.strategy_factory
        param_space = self.param_space
        settings = self.settings

        def objective(trial) -> float:
            params = _sample_params(trial, param_space)
            strategy = strategy_factory(params)
            sharpe, _ = _run_backtest_sync(strategy, train_bars, settings)
            return sharpe if sharpe is not None else -999.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_sharpe = study.best_value
        return best_params, float(best_sharpe)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_bars(bars: list[Bar], n: int) -> list[list[Bar]]:
    """Split bars into n roughly equal segments."""
    size = len(bars) // n
    segments = []
    for i in range(n):
        start = i * size
        end = start + size if i < n - 1 else len(bars)
        segments.append(bars[start:end])
    return segments


def _sample_params(trial, param_space: Dict[str, Dict]) -> Dict[str, Any]:
    """Sample params from Optuna trial according to the space spec."""
    params = {}
    for name, spec in param_space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            log = spec.get("log", False)
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=log)
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def _run_backtest_sync(
    strategy: Strategy,
    bars: list[Bar],
    settings: Settings,
) -> tuple[Optional[float], Optional[float]]:
    """
    Fast synchronous backtest for optimization trials.

    Returns (sharpe_ratio, total_return_pct). Uses simplified fill logic
    (no async OMS) for speed. Risk limits are not enforced — this is a
    signal-quality proxy, not a production simulation.
    """
    strategy.reset()
    capital = settings.strategy.initial_capital
    cash = capital
    position = 0  # shares held

    equity: list[float] = [capital]

    for bar in bars:
        event = MarketDataEvent(bar=bar)
        signal = strategy.on_market_data(event)

        if signal is not None:
            weight = settings.strategy.strategy_weights.get(signal.strategy_id, 0.33)
            target_value = capital * weight * signal.confidence

            if signal.direction > 0 and position == 0:
                shares = math.floor(min(target_value, cash * 0.95) / bar.close)
                if shares > 0:
                    position = shares
                    cash -= shares * bar.close
            elif signal.direction <= 0 and position > 0:
                cash += position * bar.close
                position = 0

        equity.append(cash + position * bar.close)

    eq = np.array(equity)
    if len(eq) < 2:
        return None, None

    total_return_pct = float((eq[-1] - eq[0]) / eq[0] * 100)

    daily_returns = np.diff(eq) / eq[:-1]
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0, total_return_pct

    sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
    return sharpe, total_return_pct
