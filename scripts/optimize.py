"""
Strategy parameter optimization entry point.

Runs walk-forward Bayesian optimization over a date range and prints
per-fold results with in-sample vs out-of-sample Sharpe comparison.

Usage:
    python scripts/optimize.py --strategy momentum --symbol AAPL \
        --start 2020-01-01 --end 2023-12-31 \
        --splits 4 --trials 50

Interpreting results:
    - In-sample Sharpe >> OOS Sharpe: overfitting — use fewer trials or
      add regularization to the param space.
    - Consistent OOS Sharpe across folds: params are robust.
    - Mean OOS Sharpe: use this (not IS) to judge strategy quality.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings, StrategySettings
from backtest.optimizer import StrategyOptimizer


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_momentum_optimizer(args, settings: Settings) -> StrategyOptimizer:
    from strategy.momentum import MomentumStrategy

    def factory(params: dict) -> MomentumStrategy:
        s = StrategySettings(
            initial_capital=settings.strategy.initial_capital,
            momentum_lookback=params["lookback"],
            momentum_threshold=params["threshold"],
        )
        return MomentumStrategy(s)

    param_space = {
        "lookback":  {"type": "int",   "low": 5,     "high": 60},
        "threshold": {"type": "float", "low": 0.003, "high": 0.05},
    }

    return StrategyOptimizer(
        strategy_factory=factory,
        param_space=param_space,
        symbol=args.symbol,
        start=parse_date(args.start),
        end=parse_date(args.end),
        settings=settings,
        n_splits=args.splits,
        n_trials=args.trials,
    )


def build_mean_reversion_optimizer(args, settings: Settings) -> StrategyOptimizer:
    from strategy.mean_reversion import MeanReversionStrategy

    def factory(params: dict) -> MeanReversionStrategy:
        s = StrategySettings(
            initial_capital=settings.strategy.initial_capital,
            mean_reversion_lookback=params["lookback"],
            mean_reversion_zscore=params["zscore"],
        )
        return MeanReversionStrategy(s)

    param_space = {
        "lookback": {"type": "int",   "low": 10,  "high": 100},
        "zscore":   {"type": "float", "low": 1.0, "high": 3.5},
    }

    return StrategyOptimizer(
        strategy_factory=factory,
        param_space=param_space,
        symbol=args.symbol,
        start=parse_date(args.start),
        end=parse_date(args.end),
        settings=settings,
        n_splits=args.splits,
        n_trials=args.trials,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy parameter optimizer")
    parser.add_argument("--strategy", choices=["momentum", "mean_reversion"], required=True)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--splits", type=int, default=4, help="Walk-forward folds")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per fold")
    args = parser.parse_args()

    settings = Settings()

    builders = {
        "momentum": build_momentum_optimizer,
        "mean_reversion": build_mean_reversion_optimizer,
    }
    optimizer = builders[args.strategy](args, settings)

    print(f"\nOptimizing {args.strategy} on {args.symbol} "
          f"({args.start} → {args.end}), "
          f"{args.splits} folds × {args.trials} trials\n")

    result = await optimizer.run()
    print(result)

    print("\n--- Overfitting check ---")
    for fold in result.folds:
        if fold.out_of_sample_sharpe is not None:
            ratio = fold.in_sample_sharpe / fold.out_of_sample_sharpe if fold.out_of_sample_sharpe != 0 else float("inf")
            flag = " *** OVERFIT ***" if ratio > 3 else ""
            print(f"  Fold {fold.fold}: IS/OOS ratio = {ratio:.1f}x{flag}")


if __name__ == "__main__":
    asyncio.run(main())
