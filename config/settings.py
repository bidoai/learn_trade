"""
Central configuration for the trading system.

All settings live here. Components receive their sub-settings via
constructor injection — nothing reads os.environ directly.

Loading:
  settings = Settings()           # reads .env automatically
  risk_engine = RiskEngine(settings.risk)

Environment variables use double-underscore nesting:
  ALPACA__API_KEY=xxx
  ALPACA__SECRET_KEY=yyy
  RISK__MAX_DAILY_LOSS_PCT=0.03

See .env.example for a full list.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlpacaSettings(BaseModel):
    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"
    data_feed: str = "iex"          # "iex" (free) or "sip" (paid)
    symbols: List[str] = ["AAPL", "GOOGL", "MSFT"]

    @field_validator("symbols")
    @classmethod
    def symbols_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("symbols list must not be empty")
        return [s.upper() for s in v]


class RiskSettings(BaseModel):
    """
    Risk limits enforced by RiskEngine.
    All percentages are expressed as decimals (0.05 = 5%).
    """
    max_position_pct: float = 0.05        # max 5% of portfolio per position
    max_daily_loss_pct: float = 0.02      # 2% daily loss → circuit breaker
    max_concentration_pct: float = 0.20   # max 20% in any single symbol
    var_confidence: float = 0.95          # 95% VaR
    var_window_days: int = 20             # rolling window for VaR
    max_var_pct: float = 0.02            # block orders if VaR > 2% of capital
    max_drawdown_pct: float = 0.10       # halt all trading if drawdown > 10%
    stop_loss_pct: float = 0.05          # per-position stop-loss at 5% loss
    limit_order_ttl_sec: int = 30        # cancel stale limit orders after 30s
    stale_data_timeout_sec: int = 60      # halt signals after 60s no data


class StrategySettings(BaseModel):
    initial_capital: float = 100_000.0
    # Momentum strategy
    momentum_lookback: int = 20
    momentum_threshold: float = 0.01     # min return to generate signal
    # Mean reversion strategy
    mean_reversion_lookback: int = 50
    mean_reversion_zscore: float = 2.0   # z-score threshold
    # ML strategy
    ml_lookback: int = 100
    ml_retrain_interval_bars: int = 500  # retrain every N bars
    ml_label_threshold: float = 0.001   # dead zone for ternary label (±0.1%)
    # Portfolio allocation weights (must sum to 1.0)
    strategy_weights: dict = {
        "momentum": 0.40,
        "mean_reversion": 0.30,
        "ml": 0.30,
    }

    @field_validator("initial_capital")
    @classmethod
    def capital_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("initial_capital must be positive")
        return v


class DashboardSettings(BaseModel):
    host: str = "127.0.0.1"    # localhost only — no external access
    port: int = 8000
    update_interval_sec: float = 1.0


class FIXSettings(BaseModel):
    """
    FIX 4.2 session configuration.

    Defaults point to the in-process simulator used in tests.
    For a real broker, override host/port and comp IDs via env vars:
        FIX__HOST=fix.broker.com FIX__PORT=9823
    """
    host: str = "127.0.0.1"
    port: int = 8888
    sender_comp_id: str = "CLIENT"
    target_comp_id: str = "SERVER"
    heartbeat_interval: int = 30


class Settings(BaseSettings):
    alpaca: AlpacaSettings
    risk: RiskSettings = RiskSettings()
    strategy: StrategySettings = StrategySettings()
    dashboard: DashboardSettings = DashboardSettings()
    fix: FIXSettings = FIXSettings()
    db_path: str = "trading.db"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )
