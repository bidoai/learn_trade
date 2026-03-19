"""
Historical market data loader.

Used by BacktestRunner to load price data for backtesting.
Falls back to yfinance if Alpaca historical data is unavailable.

Returns an iterator of Bar objects sorted by timestamp (oldest first).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterator

import pandas as pd
import structlog

from core.models import Bar

logger = structlog.get_logger(__name__)


class HistoricalLoader:
    def __init__(self, api_key: str = "", secret_key: str = "") -> None:
        self.api_key = api_key
        self.secret_key = secret_key

    def load(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1Day",
    ) -> list[Bar]:
        """
        Load historical bars for a symbol.
        Tries Alpaca first, falls back to yfinance.
        Raises ValueError if no data is available.
        """
        try:
            return self._load_alpaca(symbol, start, end, timeframe)
        except Exception as e:
            logger.warning(
                "historical.alpaca_failed",
                symbol=symbol,
                error=str(e),
                fallback="yfinance",
            )
            return self._load_yfinance(symbol, start, end)

    def _load_alpaca(self, symbol: str, start: date, end: date, timeframe: str) -> list[Bar]:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.min.time()),
        )
        data = client.get_stock_bars(request)
        df = data.df

        if df.empty:
            raise ValueError(f"No Alpaca data for {symbol}")

        return self._df_to_bars(df, symbol)

    def _load_yfinance(self, symbol: str, start: date, end: date) -> list[Bar]:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed: pip install yfinance")

        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No yfinance data for {symbol} between {start} and {end}")

        return self._df_to_bars(df, symbol)

    def _df_to_bars(self, df: pd.DataFrame, symbol: str) -> list[Bar]:
        bars = []
        df = df.sort_index()

        for ts, row in df.iterrows():
            try:
                # Handle both multi-index (Alpaca) and single index (yfinance)
                timestamp = ts[1] if isinstance(ts, tuple) else ts
                if hasattr(timestamp, "to_pydatetime"):
                    timestamp = timestamp.to_pydatetime()

                bar = Bar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(row.get("open", row.get("Open", 0))),
                    high=float(row.get("high", row.get("High", 0))),
                    low=float(row.get("low", row.get("Low", 0))),
                    close=float(row.get("close", row.get("Close", 0))),
                    volume=int(row.get("volume", row.get("Volume", 0))),
                )
                if bar.is_valid():
                    bars.append(bar)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("historical.bar_parse_error", symbol=symbol, error=str(e))
                continue

        logger.info("historical.loaded", symbol=symbol, bars=len(bars))
        return bars
