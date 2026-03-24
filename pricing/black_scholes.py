"""
Black-Scholes options pricing and Greeks.

Pure math module — no dependencies on the rest of the trading system.
All functions operate on scalar floats. Use scipy.stats.norm for N() and N'().

Conventions:
  S     = current underlying price
  K     = strike price
  r     = annual risk-free rate (e.g. 0.05 = 5%)
  T     = time to expiry in years (e.g. 0.5 = 6 months)
  sigma = annualized volatility (e.g. 0.20 = 20%)
"""
from __future__ import annotations

import math

from scipy.stats import norm


def _check_sigma(sigma: float) -> None:
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")


def d1(S: float, K: float, r: float, T: float, sigma: float) -> float:
    _check_sigma(sigma)
    return (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, r: float, T: float, sigma: float) -> float:
    _check_sigma(sigma)
    return d1(S, K, r, T, sigma) - sigma * math.sqrt(T)


def call_price(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    _check_sigma(sigma)
    _d1 = d1(S, K, r, T, sigma)
    _d2 = d2(S, K, r, T, sigma)
    return S * norm.cdf(_d1) - K * math.exp(-r * T) * norm.cdf(_d2)


def put_price(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    _check_sigma(sigma)
    # put-call parity: P = C - S + K·e^{-rT}
    return call_price(S, K, r, T, sigma) - S + K * math.exp(-r * T)


def delta(S: float, K: float, r: float, T: float, sigma: float, option_type: str = "call") -> float:
    if T <= 0:
        return 0.0
    _check_sigma(sigma)
    _d1 = d1(S, K, r, T, sigma)
    if option_type == "call":
        return norm.cdf(_d1)
    elif option_type == "put":
        return norm.cdf(_d1) - 1.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def gamma(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    _check_sigma(sigma)
    _d1 = d1(S, K, r, T, sigma)
    return norm.pdf(_d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    _check_sigma(sigma)
    _d1 = d1(S, K, r, T, sigma)
    return S * norm.pdf(_d1) * math.sqrt(T)


def theta(S: float, K: float, r: float, T: float, sigma: float, option_type: str = "call") -> float:
    if T <= 0:
        return 0.0
    _check_sigma(sigma)
    _d1 = d1(S, K, r, T, sigma)
    _d2 = d2(S, K, r, T, sigma)
    common = -S * norm.pdf(_d1) * sigma / (2 * math.sqrt(T))
    discount = K * math.exp(-r * T)
    if option_type == "call":
        annualized = common - r * discount * norm.cdf(_d2)
    elif option_type == "put":
        annualized = common + r * discount * norm.cdf(-_d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return annualized / 252


def rho(S: float, K: float, r: float, T: float, sigma: float, option_type: str = "call") -> float:
    if T <= 0:
        return 0.0
    _check_sigma(sigma)
    _d2 = d2(S, K, r, T, sigma)
    discount = K * T * math.exp(-r * T)
    if option_type == "call":
        return discount * norm.cdf(_d2) / 100
    elif option_type == "put":
        return -discount * norm.cdf(-_d2) / 100
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson implied volatility solver."""
    if T <= 0:
        raise ValueError("Cannot compute implied vol at expiry (T <= 0)")

    pricer = call_price if option_type == "call" else put_price

    # Initial guess: simple approximation
    sigma = 0.20

    for _ in range(max_iter):
        price = pricer(S, K, r, T, sigma)
        v = vega(S, K, r, T, sigma)
        if abs(v) < 1e-12:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / v
        if sigma <= 0:
            sigma = 1e-6

    # Final convergence check
    price = pricer(S, K, r, T, sigma)
    if abs(price - market_price) < tol:
        return sigma

    raise ValueError(
        f"implied_vol did not converge after {max_iter} iterations. "
        f"market_price={market_price}, final_price={price:.6f}, sigma={sigma:.6f}"
    )
