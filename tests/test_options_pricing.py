"""Tests for pricing/black_scholes.py"""
from __future__ import annotations

import math

import pytest

from pricing.black_scholes import (
    call_price,
    delta,
    gamma,
    implied_vol,
    put_price,
    rho,
    theta,
    vega,
)


class TestBlackScholes:
    def test_call_put_parity(self):
        """C - P = S - K·e^{-rT} (put-call parity)"""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20
        C = call_price(S, K, r, T, sigma)
        P = put_price(S, K, r, T, sigma)
        assert abs(C - P - (S - K * math.exp(-r * T))) < 1e-8

    def test_call_price_known_value(self):
        """Known Black-Scholes call value: S=100, K=100, r=0.05, T=1, σ=0.20 → ~10.45"""
        price = call_price(100, 100, 0.05, 1.0, 0.20)
        assert abs(price - 10.4506) < 0.01

    def test_delta_call_between_0_and_1(self):
        d = delta(100, 100, 0.05, 1.0, 0.20, "call")
        assert 0 < d < 1

    def test_delta_put_between_neg1_and_0(self):
        d = delta(100, 100, 0.05, 1.0, 0.20, "put")
        assert -1 < d < 0

    def test_delta_call_plus_put_equals_1(self):
        """delta_call - delta_put = 1"""
        S, K, r, T, sigma = 100, 110, 0.05, 0.5, 0.25
        dc = delta(S, K, r, T, sigma, "call")
        dp = delta(S, K, r, T, sigma, "put")
        assert abs(dc - dp - 1.0) < 1e-10

    def test_gamma_positive(self):
        g = gamma(100, 100, 0.05, 1.0, 0.20)
        assert g > 0

    def test_vega_positive(self):
        v = vega(100, 100, 0.05, 1.0, 0.20)
        assert v > 0

    def test_theta_negative_for_long_call(self):
        """Long options lose time value daily (theta < 0)."""
        t = theta(100, 100, 0.05, 1.0, 0.20, "call")
        assert t < 0

    def test_implied_vol_roundtrip(self):
        """Imply vol from a BS price; should recover the original vol."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.25
        price = call_price(S, K, r, T, sigma)
        iv = implied_vol(price, S, K, r, T, "call")
        assert abs(iv - sigma) < 1e-5

    def test_intrinsic_value_at_expiry(self):
        """T=0 call should return intrinsic value."""
        assert abs(call_price(110, 100, 0.05, 0, 0.20) - 10.0) < 1e-8
        assert abs(put_price(90, 100, 0.05, 0, 0.20) - 10.0) < 1e-8

    def test_deep_itm_call_delta_near_1(self):
        d = delta(200, 100, 0.05, 1.0, 0.20, "call")
        assert d > 0.99

    def test_deep_otm_call_delta_near_0(self):
        d = delta(50, 100, 0.05, 1.0, 0.20, "call")
        assert d < 0.01

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            call_price(100, 100, 0.05, 1.0, 0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            delta(100, 100, 0.05, 1.0, -0.1, "call")

    def test_rho_call_positive(self):
        """Call rho is positive — higher rates increase call value."""
        r_val = rho(100, 100, 0.05, 1.0, 0.20, "call")
        assert r_val > 0

    def test_rho_put_negative(self):
        """Put rho is negative — higher rates decrease put value."""
        r_val = rho(100, 100, 0.05, 1.0, 0.20, "put")
        assert r_val < 0
