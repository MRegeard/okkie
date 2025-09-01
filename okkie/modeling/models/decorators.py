from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache, wraps
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_RT2 = np.sqrt(2.0)
_PI = np.pi


def _as_scalar(x) -> float:
    """Accept scalar or length-1 array; return float."""
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("edge must be scalar or length-1 array")
    return float(a[0])


@lru_cache(maxsize=128)
def _shifts(period: float, truncation: int) -> np.ndarray:
    """k*period for k in [-K..K]."""
    P = float(period)
    K = int(truncation)
    return np.arange(-K, K + 1, dtype=float) * P


class VectorizeIntegral:
    """
    Decorator class for integral functions to:
      - read `edge_min`/`edge_max` as scalars (or length-1 arrays),
      - broadcast specified parameter kwargs to a common shape,
      - optionally clamp parameters (e.g. sigma >= 1e-300),
      - return zeros with broadcasted shape if `edge_max <= edge_min`,
      - reshape the flat result back to the broadcast shape.

    The wrapped function must accept:
      func(a: float, b: float, **kwargs_arrays) -> 1D array (flat),
    where kwargs_arrays contain the broadcasted, flattened arrays for the
    names listed in `params_to_broadcast`. All other kwargs are passed through.
    """

    def __init__(
        self,
        params_to_broadcast: Iterable[str],
        *,
        clips: Mapping[str, tuple[float | None, float | None]] | None = None,
        return_zeros_if_empty_interval: bool = True,
    ):
        self.params = tuple(params_to_broadcast)
        self.clips: dict[str, tuple[float | None, float | None]] = dict(clips or {})
        self.zero_if_empty = bool(return_zeros_if_empty_interval)

    def __call__(self, func):
        @wraps(func)
        def wrapper(edge_min, edge_max, **kwargs):
            # edges -> scalars
            a = _as_scalar(edge_min)
            b = _as_scalar(edge_max)

            # collect arrays to broadcast
            try:
                arrays = [np.asarray(kwargs[name], dtype=float) for name in self.params]
            except KeyError as e:
                raise KeyError(
                    f"Missing required parameter for vectorized integral: {e}"
                )

            # broadcast to common shape
            bcast = np.broadcast_arrays(*arrays)
            shp = bcast[0].shape  # output shape

            # apply clips (if any) and flatten
            for name, arr in zip(self.params, bcast):
                lo, hi = self.clips.get(name, (None, None))
                if lo is not None or hi is not None:
                    lo_val = -np.inf if lo is None else float(lo)
                    hi_val = np.inf if hi is None else float(hi)
                    arr = np.clip(arr, lo_val, hi_val)
                kwargs[name] = arr.reshape(-1)

            # early return zeros if interval empty
            if self.zero_if_empty and not (b > a):
                return np.zeros(shp, dtype=float)

            # call the core implementation (expects flat arrays) and reshape
            out = np.asarray(func(a, b, **kwargs), dtype=float)
            if out.ndim != 1 or out.size != np.prod(shp, dtype=int):
                raise ValueError(
                    f"{func.__name__} must return a 1D array of length {np.prod(shp)}; got shape {out.shape}"
                )
            return out.reshape(shp)

        return wrapper
