from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import erf

__all__ = [
    "integrate_asymm_gaussian",
    "integrate_asymm_lorentzian",
    "integrate_gaussian",
    "integrate_lorentzian",
    "integrate_periodic_asymm_gaussian",
    "integrate_periodic_asymm_lorentzian",
    "integrate_periodic_gaussian",
    "integrate_periodic_lorentzian",
    "integrate_trapezoid",
]


@lru_cache(maxsize=64)
def _shifts(period: float, truncation: int) -> np.ndarray:
    """Periodic shift grid: k*period for k in [-K..K]."""
    K = int(truncation)
    return np.arange(-K, K + 1, dtype=float) * float(period)


_RT2 = np.sqrt(2.0)
_PI = np.pi


def integrate_trapezoid(
    model, edge_min: float, edge_max: float, n: int = 2048
) -> float:
    """Numerical fallback (trapezoid) for arbitrary PhaseModel."""
    a = float(edge_min)
    b = float(edge_max)
    if b <= a:
        return 0.0
    x = np.linspace(a, b, n, dtype=float)
    y = np.asarray(model(x), float)
    # np.trapz is already vectorized C code
    return float(np.trapezoid(y, x))


def integrate_gaussian(edge_min, edge_max, *, amplitude, mean, sigma) -> float:
    a = float(edge_min)
    b = float(edge_max)
    if b <= a:
        return 0.0
    s = float(sigma)
    mu = float(mean)
    A = (a - mu) / (_RT2 * s)
    B = (b - mu) / (_RT2 * s)
    return float(amplitude * s * np.sqrt(_PI / 2.0) * (erf(B) - erf(A)))


def integrate_lorentzian(edge_min, edge_max, *, amplitude, mean, sigma) -> float:
    a = float(edge_min)
    b = float(edge_max)
    if b <= a:
        return 0.0
    s = float(sigma)
    mu = float(mean)
    A = (a - mu) / s
    B = (b - mu) / s
    return float(amplitude * s * (np.arctan(B) - np.arctan(A)))


def integrate_asymm_gaussian(
    edge_min, edge_max, *, amplitude, mean, sigma_1, sigma_2
) -> float:
    """Asymmetric Gaussian (simga_1 on left of mean, sigma_2 on right)."""
    a = float(edge_min)
    b = float(edge_max)
    if b <= a:
        return 0.0
    c = float(mean)
    s1 = float(sigma_1)
    s2 = float(sigma_2)

    # left part: [a, min(b, c)] with simga_1
    left_hi = min(b, c)
    left = 0.0
    if a < c:
        A = (a - c) / (_RT2 * s1)
        H = (left_hi - c) / (_RT2 * s1)
        left = s1 * np.sqrt(_PI / 2.0) * (erf(H) - erf(A))

    # right part: [max(a, c), b] with simga_2
    right_lo = max(a, c)
    right = 0.0
    if b > c:
        L = (right_lo - c) / (_RT2 * s2)
        B = (b - c) / (_RT2 * s2)
        right = s2 * np.sqrt(_PI / 2.0) * (erf(B) - erf(L))

    return float(amplitude * (left + right))


def integrate_asymm_lorentzian(
    edge_min, edge_max, *, amplitude, mean, sigma_1, sigma_2
) -> float:
    """Asymmetric Lorentzian (sigma_1 left of mean, sigma_2 right)."""
    a = float(edge_min)
    b = float(edge_max)
    if b <= a:
        return 0.0
    c = float(mean)
    s1 = float(sigma_1)
    s2 = float(sigma_2)

    # left: [a, min(b, c)] with sigma_1
    left_hi = min(b, c)
    left = 0.0
    if a < c:
        A = (a - c) / s1
        H = (left_hi - c) / s1
        left = s1 * (np.arctan(H) - np.arctan(A))

    # right: [max(a, c), b] with sigma_2
    right_lo = max(a, c)
    right = 0.0
    if b > c:
        L = (right_lo - c) / s2
        B = (b - c) / s2
        right = s2 * (np.arctan(B) - np.arctan(L))

    return float(amplitude * (left + right))


def integrate_periodic_gaussian(
    edge_min, edge_max, *, amplitude, mean, sigma, period, truncation
) -> float:
    """formula"""
    a = float(edge_min)
    b = float(edge_max)
    if b <= a:
        return 0.0
    s = float(sigma)
    mu = float(mean)
    P = float(period)
    shifts = _shifts(P, int(truncation))
    A = (a - mu + shifts) / (_RT2 * s)
    B = (b - mu + shifts) / (_RT2 * s)
    return float(amplitude * s * np.sqrt(_PI / 2.0) * np.sum(erf(B) - erf(A)))


def integrate_periodic_lorentzian(
    edge_min, edge_max, *, amplitude, mean, sigma, period, truncation
) -> float:
    """Formula"""
    a = float(edge_min)
    b = float(edge_max)
    if b <= a:
        return 0.0
    s = float(sigma)
    mu = float(mean)
    P = float(period)
    shifts = _shifts(P, int(truncation))
    A = (a - mu + shifts) / s
    B = (b - mu + shifts) / s
    return float(amplitude * s * np.sum(np.arctan(B) - np.arctan(A)))


def integrate_periodic_asymm_gaussian(
    edge_min, edge_max, *, amplitude, mean, sigma_1, sigma_2, period, truncation
) -> float:
    """
    Wrapped asymmetric Gaussian:
      left of center (x < c_k) uses σ1; right (x >= c_k) uses σ2, where c_k = mean - kP.
    Integral is sum over k, splitting each interval at c_k if it lies inside [a,b].
    """
    a = float(edge_min)
    b = float(edge_max)
    if b <= a:
        return 0.0
    mu = float(mean)
    s1 = float(sigma_1)
    s2 = float(sigma_2)
    P = float(period)
    shifts = _shifts(P, int(truncation))
    centers = mu - shifts

    left_hi = np.minimum(b, centers)
    left_mask = a < centers
    left = np.zeros_like(centers, dtype=float)
    if np.any(left_mask):
        A = (a - centers[left_mask]) / (_RT2 * s1)
        H = (left_hi[left_mask] - centers[left_mask]) / (_RT2 * s1)
        left[left_mask] = s1 * np.sqrt(_PI / 2.0) * (erf(H) - erf(A))

    right_lo = np.maximum(a, centers)
    right_mask = b > centers
    right = np.zeros_like(centers, dtype=float)
    if np.any(right_mask):
        L = (right_lo[right_mask] - centers[right_mask]) / (_RT2 * s2)
        B = (b - centers[right_mask]) / (_RT2 * s2)
        right[right_mask] = s2 * np.sqrt(_PI / 2.0) * (erf(B) - erf(L))

    return float(amplitude * (left + right).sum())


def integrate_periodic_asymm_lorentzian(
    edge_min, edge_max, *, amplitude, mean, sigma_1, sigma_2, period, truncation
) -> float:
    """
    Wrapped asymmetric Lorentzian:
      left of center uses σ1; right uses σ2, with centers c_k = mean - kP.
    """
    a = float(edge_min)
    b = float(edge_max)
    if b <= a:
        return 0.0
    mu = float(mean)
    s1 = float(sigma_1)
    s2 = float(sigma_2)
    P = float(period)
    shifts = _shifts(P, int(truncation))
    centers = mu - shifts

    # Left segments
    left_hi = np.minimum(b, centers)
    left_mask = a < centers
    left = np.zeros_like(centers, dtype=float)
    if np.any(left_mask):
        A = (a - centers[left_mask]) / s1
        H = (left_hi[left_mask] - centers[left_mask]) / s1
        left[left_mask] = s1 * (np.arctan(H) - np.arctan(A))

    # Right segments
    right_lo = np.maximum(a, centers)
    right_mask = b > centers
    right = np.zeros_like(centers, dtype=float)
    if np.any(right_mask):
        L = (right_lo[right_mask] - centers[right_mask]) / s2
        B = (b - centers[right_mask]) / s2
        right[right_mask] = s2 * (np.arctan(B) - np.arctan(L))

    return float(amplitude * (left + right).sum())
