from __future__ import annotations

import numpy as np
from scipy.special import erf

from .decorators import VectorizeIntegral, _as_scalar, _shifts

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


_RT2 = np.sqrt(2.0)
_PI = np.pi
_SQRT_PI_OVER_2 = np.sqrt(_PI / 2.0)


def integrate_trapezoid(model, edge_min, edge_max, n: int = 2048) -> float:
    a = _as_scalar(edge_min)
    b = _as_scalar(edge_max)
    if b <= a:
        return 0.0
    x = np.linspace(a, b, n, dtype=float)
    y = np.asarray(model(x), float)
    return float(np.trapezoid(y, x))


@VectorizeIntegral(
    params_to_broadcast=("amplitude", "mean", "sigma"), clips={"sigma": (1e-300, None)}
)
def integrate_gaussian(a, b, *, amplitude, mean, sigma):
    # amplitude, mean, sigma are 1D arrays (flat) of same length
    AA = (a - mean) / (_RT2 * sigma)
    BB = (b - mean) / (_RT2 * sigma)
    return amplitude * sigma * _SQRT_PI_OVER_2 * (erf(BB) - erf(AA))


@VectorizeIntegral(
    params_to_broadcast=("amplitude", "mean", "sigma"), clips={"sigma": (1e-300, None)}
)
def integrate_lorentzian(a, b, *, amplitude, mean, sigma):
    AA = (a - mean) / sigma
    BB = (b - mean) / sigma
    return amplitude * sigma * (np.arctan(BB) - np.arctan(AA))


@VectorizeIntegral(
    params_to_broadcast=("amplitude", "mean", "sigma_1", "sigma_2"),
    clips={"sigma_1": (1e-300, None), "sigma_2": (1e-300, None)},
)
def integrate_asymm_gaussian(a, b, *, amplitude, mean, sigma_1, sigma_2):
    C = mean
    # left segment [a, min(b, C)] uses sigma_1
    left_hi = np.minimum(b, C)
    left_mask = (a < C).astype(float)
    A1 = (a - C) / (_RT2 * sigma_1)
    H1 = (left_hi - C) / (_RT2 * sigma_1)
    left = left_mask * (sigma_1 * _SQRT_PI_OVER_2 * (erf(H1) - erf(A1)))
    # right segment [max(a, C), b] uses sigma_2
    right_lo = np.maximum(a, C)
    right_mask = (b > C).astype(float)
    L2 = (right_lo - C) / (_RT2 * sigma_2)
    B2 = (b - C) / (_RT2 * sigma_2)
    right = right_mask * (sigma_2 * _SQRT_PI_OVER_2 * (erf(B2) - erf(L2)))
    return amplitude * (left + right)


@VectorizeIntegral(
    params_to_broadcast=("amplitude", "mean", "sigma_1", "sigma_2"),
    clips={"sigma_1": (1e-300, None), "sigma_2": (1e-300, None)},
)
def integrate_asymm_lorentzian(a, b, *, amplitude, mean, sigma_1, sigma_2):
    C = mean
    left_hi = np.minimum(b, C)
    left_mask = (a < C).astype(float)
    A1 = (a - C) / sigma_1
    H1 = (left_hi - C) / sigma_1
    left = left_mask * (sigma_1 * (np.arctan(H1) - np.arctan(A1)))
    right_lo = np.maximum(a, C)
    right_mask = (b > C).astype(float)
    L2 = (right_lo - C) / sigma_2
    B2 = (b - C) / sigma_2
    right = right_mask * (sigma_2 * (np.arctan(B2) - np.arctan(L2)))
    return amplitude * (left + right)


@VectorizeIntegral(
    params_to_broadcast=("amplitude", "mean", "sigma"), clips={"sigma": (1e-300, None)}
)
def integrate_periodic_gaussian(a, b, *, amplitude, mean, sigma, period, truncation):
    P = float(period)
    K = int(truncation)
    shifts = _shifts(P, K)
    # Build (N, M) arguments via broadcasting
    Aarg = (a - mean[:, None] + shifts[None, :]) / (_RT2 * sigma[:, None])
    Barg = (b - mean[:, None] + shifts[None, :]) / (_RT2 * sigma[:, None])
    # sum across images
    integ = (
        amplitude[:, None]
        * (sigma[:, None] * _SQRT_PI_OVER_2)
        * (erf(Barg) - erf(Aarg))
    )
    return integ.sum(axis=1)


@VectorizeIntegral(
    params_to_broadcast=("amplitude", "mean", "sigma"), clips={"sigma": (1e-300, None)}
)
def integrate_periodic_lorentzian(a, b, *, amplitude, mean, sigma, period, truncation):
    P = float(period)
    K = int(truncation)
    shifts = _shifts(P, K)
    Aarg = (a - mean[:, None] + shifts[None, :]) / (sigma[:, None])
    Barg = (b - mean[:, None] + shifts[None, :]) / (sigma[:, None])
    integ = amplitude[:, None] * sigma[:, None] * (np.arctan(Barg) - np.arctan(Aarg))
    return integ.sum(axis=1)


@VectorizeIntegral(
    params_to_broadcast=("amplitude", "mean", "sigma_1", "sigma_2"),
    clips={"sigma_1": (1e-300, None), "sigma_2": (1e-300, None)},
)
def integrate_periodic_asymm_gaussian(
    a, b, *, amplitude, mean, sigma_1, sigma_2, period, truncation
):
    P = float(period)
    K = int(truncation)
    shifts = _shifts(P, K)
    centers = mean[:, None] - shifts[None, :]

    # Left of center uses sigma_1
    left_hi = np.minimum(b, centers)
    left_mask = (a < centers).astype(float)
    A1 = (a - centers) / (_RT2 * sigma_1[:, None])
    H1 = (left_hi - centers) / (_RT2 * sigma_1[:, None])
    left = left_mask * (sigma_1[:, None] * _SQRT_PI_OVER_2 * (erf(H1) - erf(A1)))

    # Right of center uses sigma_2
    right_lo = np.maximum(a, centers)
    right_mask = (b > centers).astype(float)
    L2 = (right_lo - centers) / (_RT2 * sigma_2[:, None])
    B2 = (b - centers) / (_RT2 * sigma_2[:, None])
    right = right_mask * (sigma_2[:, None] * _SQRT_PI_OVER_2 * (erf(B2) - erf(L2)))

    return (amplitude[:, None] * (left + right)).sum(axis=1)


@VectorizeIntegral(
    params_to_broadcast=("amplitude", "mean", "sigma_1", "sigma_2"),
    clips={"sigma_1": (1e-300, None), "sigma_2": (1e-300, None)},
)
def integrate_periodic_asymm_lorentzian(
    a, b, *, amplitude, mean, sigma_1, sigma_2, period, truncation
):
    P = float(period)
    K = int(truncation)
    shifts = _shifts(P, K)
    centers = mean[:, None] - shifts[None, :]

    left_hi = np.minimum(b, centers)
    left_mask = (a < centers).astype(float)
    A1 = (a - centers) / (sigma_1[:, None])
    H1 = (left_hi - centers) / (sigma_1[:, None])
    left = left_mask * (sigma_1[:, None] * (np.arctan(H1) - np.arctan(A1)))

    right_lo = np.maximum(a, centers)
    right_mask = (b > centers).astype(float)
    L2 = (right_lo - centers) / (sigma_2[:, None])
    B2 = (b - centers) / (sigma_2[:, None])
    right = right_mask * (sigma_2[:, None] * (np.arctan(B2) - np.arctan(L2)))

    return (amplitude[:, None] * (left + right)).sum(axis=1)
