from __future__ import annotations

import inspect
from functools import lru_cache, wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

import numpy as np

_RT2 = np.sqrt(2.0)
_PI = np.pi


def _as_scalar(x) -> float:
    """Accept scalar or length-1 array; return float."""
    a = np.asarray(x, dtype=float).ravel()
    if a.size != 1:
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


class VectorizeEvaluate:
    """
    Decorator class for `evaluate` methods to handle:
      - Outer broadcasting between `phase` (shape P...) and model parameters (shape Q...).
      - Modulo-period mapping for chosen parameter names (e.g. 'mean').
      - Optional clipping for chosen parameter names (e.g. sigma >= 1e-300).
      - (Optional) Providing cached periodic `shifts` (k*period) to the core function.
      - Returning a result with shape P... + Q....

    Usage pattern (core `evaluate` becomes very small):

        @VectorizeEvaluate(
            params_to_broadcast=("mean", "sigma"),
            mod_by_period=("mean",),
            clips={"sigma": (1e-300, None)},
            provide_shifts=True,
        )
        def evaluate(phase, mean, sigma, period, wrapping_truncation, *, shifts=None, _vmeta=None):
            # phase, mean, sigma are already outer-shaped and broadcast:
            #   phase : P... x 1... (Qnd)
            #   mean  : 1... (Pnd) x Q...
            #   sigma : 1... (Pnd) x Q...
            # _vmeta = {"phase_ndim": Pnd, "param_shape": Qshp}
            Pnd = _vmeta["phase_ndim"]; Qshp = _vmeta["param_shape"]

            delta = (phase - mean)[..., None] + shifts  # add wrapped shift axis
            num = np.exp(-0.5 * (delta / sigma[..., None])**2).sum(axis=-1)  # -> P... x Q...
            return num  # (if normalized elsewhere), or divide by your (reshaped) denominator here.

    Parameters
    ----------
    params_to_broadcast : iterable[str]
        Names of model parameters to broadcast in the "Q..." group (e.g. ("mean","sigma")).
    mod_by_period : iterable[str], optional
        Subset of `params_to_broadcast` to map with modulo `period` (e.g. ("mean",)).
    clips : dict[str, tuple[lo, hi]] | None
        Optional clipping per parameter name. Use None to skip a bound (e.g. {"sigma": (1e-300, None)}).
    provide_shifts : bool
        If True, computes and passes `shifts` (k*period) to the core function (kwarg).
        The core function should accept `shifts=None` if it doesn't need it.

    The wrapper passes a meta dict `_vmeta={"phase_ndim": Pnd, "param_shape": Qshp}` as a kwarg.
    """

    def __init__(
        self,
        params_to_broadcast: Iterable[str],
        *,
        mod_by_period: Iterable[str] = (),
        clips: Mapping[str, tuple[float | None, float | None]] | None = None,
        provide_shifts: bool = True,
    ):
        self.params = tuple(params_to_broadcast)
        self.modset = set(mod_by_period or ())
        self.clips: dict[str, tuple[float | None, float | None]] = dict(clips or {})
        self.provide_shifts = bool(provide_shifts)

    def __call__(self, func):
        sig = inspect.signature(func)
        names = list(sig.parameters.keys())
        if len(names) < 4:
            raise TypeError(
                "evaluate must have at least (phase, ..., period, wrapping_truncation) in its signature."
            )
        if "period" not in names or "wrapping_truncation" not in names:
            raise TypeError(
                "evaluate must have 'period' and 'wrapping_truncation' parameters."
            )

        # sanity: all broadcast params must be in signature
        for p in self.params:
            if p not in names:
                raise TypeError(
                    f"Parameter '{p}' not found in evaluate signature {names}"
                )

        accepts_shifts = "shifts" in names
        accepts_meta = "_vmeta" in names

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Extract key inputs
            phase = np.asarray(bound.arguments["phase"], dtype=float)
            P = float(bound.arguments["period"])
            K = int(bound.arguments["wrapping_truncation"])

            # Collect parameters to broadcast as arrays
            parr = [np.asarray(bound.arguments[p], dtype=float) for p in self.params]
            if parr:
                parr = np.broadcast_arrays(*parr)
                Qshp = parr[0].shape
                Qnd = parr[0].ndim
            else:
                Qshp = ()
                Qnd = 0

            Pshp = phase.shape
            Pnd = phase.ndim

            # Prepare outer-shaped phase and params
            ph_outer = np.mod(phase, P).reshape(Pshp + (1,) * Qnd)

            # Write back transformed params
            for name, arr in zip(self.params, parr):
                arr = np.mod(arr, P) if name in self.modset else arr
                lo, hi = self.clips.get(name, (None, None))
                if lo is not None or hi is not None:
                    lo_val = -np.inf if lo is None else float(lo)
                    hi_val = np.inf if hi is None else float(hi)
                    arr = np.clip(arr, lo_val, hi_val)
                bound.arguments[name] = arr.reshape((1,) * Pnd + Qshp)

            # Overwrite phase with outer-shaped version
            bound.arguments["phase"] = ph_outer

            # Optionally add shifts
            if accepts_shifts and self.provide_shifts:
                bound.arguments["shifts"] = _shifts(P, K)

            # Optional meta
            if accepts_meta:
                bound.arguments["_vmeta"] = {"phase_ndim": Pnd, "param_shape": Qshp}

            # Call core function (it should return something broadcastable to P...xQ...)
            out = func(**bound.arguments)

            # Final shape = P... x Q...
            target_shape = Pshp + Qshp
            out = np.asarray(out, dtype=float)
            try:
                out = np.broadcast_to(out, target_shape)
            except Exception as e:
                raise ValueError(
                    f"{func.__name__} must return array broadcastable to {target_shape}, got {out.shape}"
                ) from e
            return out

        return wrapper
