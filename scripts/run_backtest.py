"""
Run a backtest.

Usage:
  python scripts/run_backtest.py --strategy momentum --symbol AAPL --start 2023-01-01 --end 2023-12-31
  python scripts/run_backtest.py --strategy mean_reversion --symbol GOOGL --start 2022-01-01 --end 2022-12-31
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from backtest.runner import BacktestRunner
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.ml_strategy import MLStrategy

STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "ml": MLStrategy,
}


async def main(args) -> None:
    settings = Settings()
    strategy_cls = STRATEGIES[args.strategy]
    strategy = strategy_cls(settings.strategy)

    runner = BacktestRunner(
        strategy=strategy,
        symbol=args.symbol.upper(),
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        settings=settings,
    )
    result = await runner.run()
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a strategy backtest")
    parser.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    asyncio.run(main(args))
