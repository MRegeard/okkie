import runpy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

_integral = runpy.run_path(
    Path(__file__).resolve().parents[1] / "okkie/modeling/models/integral.py"
)
integrate_asymm_gaussian = _integral["integrate_asymm_gaussian"]
integrate_asymm_lorentzian = _integral["integrate_asymm_lorentzian"]
integrate_gaussian = _integral["integrate_gaussian"]
integrate_lorentzian = _integral["integrate_lorentzian"]
integrate_periodic_asymm_gaussian = _integral["integrate_periodic_asymm_gaussian"]
integrate_periodic_asymm_lorentzian = _integral["integrate_periodic_asymm_lorentzian"]
integrate_periodic_gaussian = _integral["integrate_periodic_gaussian"]
integrate_periodic_lorentzian = _integral["integrate_periodic_lorentzian"]
integrate_trapezoid = _integral["integrate_trapezoid"]


def _numeric_trapz(func, a, b, n=10000):
    x = np.linspace(a, b, n)
    y = func(x)
    return np.trapezoid(y, x)


def test_integrate_trapezoid_matches_gaussian():
    a, b = 0.0, 1.0
    pars = dict(amplitude=10.0, mean=0.3, sigma=0.1)

    def gaussian(x):
        return pars["amplitude"] * np.exp(-0.5 * ((x - pars["mean"]) / pars["sigma"]) ** 2)

    numeric = integrate_trapezoid(gaussian, a, b, n=20000)
    analytic = integrate_gaussian(a, b, **pars)
    assert_allclose(numeric, analytic, rtol=1e-3)


@pytest.mark.parametrize(
    "func,pars,model",
    [
        (
            integrate_gaussian,
            dict(amplitude=4.0, mean=0.2, sigma=0.15),
            lambda x, p: p["amplitude"]
            * np.exp(-0.5 * ((x - p["mean"]) / p["sigma"]) ** 2),
        ),
        (
            integrate_lorentzian,
            dict(amplitude=5.0, mean=0.4, sigma=0.1),
            lambda x, p: p["amplitude"]
            / (1.0 + ((x - p["mean"]) / p["sigma"]) ** 2),
        ),
    ],
)
def test_basic_shapes(func, pars, model):
    a, b = 0.0, 1.0
    numeric = _numeric_trapz(lambda x: model(x, pars), a, b)
    analytic = func(a, b, **pars)
    assert_allclose(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize(
    "func,pars,model",
    [
        (
            integrate_asymm_gaussian,
            dict(amplitude=6.0, mean=0.5, sigma_1=0.1, sigma_2=0.2),
            lambda x, p: p["amplitude"]
            * np.where(
                x < p["mean"],
                np.exp(-0.5 * ((x - p["mean"]) / p["sigma_1"]) ** 2),
                np.exp(-0.5 * ((x - p["mean"]) / p["sigma_2"]) ** 2),
            ),
        ),
        (
            integrate_asymm_lorentzian,
            dict(amplitude=8.0, mean=0.3, sigma_1=0.05, sigma_2=0.15),
            lambda x, p: p["amplitude"]
            * np.where(
                x < p["mean"],
                1.0 / (1.0 + ((x - p["mean"]) / p["sigma_1"]) ** 2),
                1.0 / (1.0 + ((x - p["mean"]) / p["sigma_2"]) ** 2),
            ),
        ),
    ],
)
def test_asymmetric_shapes(func, pars, model):
    a, b = 0.0, 1.0
    numeric = _numeric_trapz(lambda x: model(x, pars), a, b)
    analytic = func(a, b, **pars)
    assert_allclose(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize(
    "func,pars,model",
    [
        (
            integrate_periodic_gaussian,
            dict(
                amplitude=2.0,
                mean=0.2,
                sigma=0.1,
                period=1.0,
                truncation=2,
            ),
            lambda x, p: p["amplitude"]
            * sum(
                np.exp(-0.5 * ((x - (p["mean"] - k * p["period"])) / p["sigma"]) ** 2)
                for k in range(-p["truncation"], p["truncation"] + 1)
            ),
        ),
        (
            integrate_periodic_lorentzian,
            dict(
                amplitude=3.0,
                mean=0.3,
                sigma=0.2,
                period=1.0,
                truncation=2,
            ),
            lambda x, p: p["amplitude"]
            * sum(
                1.0
                / (1.0 + ((x - (p["mean"] - k * p["period"])) / p["sigma"]) ** 2)
                for k in range(-p["truncation"], p["truncation"] + 1)
            ),
        ),
    ],
)
def test_periodic_shapes(func, pars, model):
    a, b = 0.0, 1.0
    numeric = _numeric_trapz(lambda x: model(x, pars), a, b)
    analytic = func(a, b, **pars)
    assert_allclose(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize(
    "func,pars,model",
    [
        (
            integrate_periodic_asymm_gaussian,
            dict(
                amplitude=7.0,
                mean=0.25,
                sigma_1=0.05,
                sigma_2=0.1,
                period=1.0,
                truncation=2,
            ),
            lambda x, p: p["amplitude"]
            * sum(
                np.where(
                    x < (p["mean"] - k * p["period"]),
                    np.exp(
                        -0.5
                        * (
                            (x - (p["mean"] - k * p["period"]))
                            / p["sigma_1"]
                        )
                        ** 2
                    ),
                    np.exp(
                        -0.5
                        * (
                            (x - (p["mean"] - k * p["period"]))
                            / p["sigma_2"]
                        )
                        ** 2
                    ),
                )
                for k in range(-p["truncation"], p["truncation"] + 1)
            ),
        ),
        (
            integrate_periodic_asymm_lorentzian,
            dict(
                amplitude=4.0,
                mean=0.6,
                sigma_1=0.05,
                sigma_2=0.2,
                period=1.0,
                truncation=2,
            ),
            lambda x, p: p["amplitude"]
            * sum(
                np.where(
                    x < (p["mean"] - k * p["period"]),
                    1.0
                    / (
                        1.0
                        + (
                            (x - (p["mean"] - k * p["period"]))
                            / p["sigma_1"]
                        )
                        ** 2
                    ),
                    1.0
                    / (
                        1.0
                        + (
                            (x - (p["mean"] - k * p["period"]))
                            / p["sigma_2"]
                        )
                        ** 2
                    ),
                )
                for k in range(-p["truncation"], p["truncation"] + 1)
            ),
        ),
    ],
)
def test_periodic_asymmetric_shapes(func, pars, model):
    a, b = 0.0, 1.0
    numeric = _numeric_trapz(lambda x: model(x, pars), a, b)
    analytic = func(a, b, **pars)
    assert_allclose(analytic, numeric, rtol=1e-6)


def test_returns_zero_if_edges_inverted():
    func_params = [
        (integrate_gaussian, dict(amplitude=1.0, mean=0.0, sigma=1.0)),
        (integrate_lorentzian, dict(amplitude=1.0, mean=0.0, sigma=1.0)),
        (
            integrate_asymm_gaussian,
            dict(amplitude=1.0, mean=0.0, sigma_1=1.0, sigma_2=2.0),
        ),
        (
            integrate_asymm_lorentzian,
            dict(amplitude=1.0, mean=0.0, sigma_1=1.0, sigma_2=2.0),
        ),
        (
            integrate_periodic_gaussian,
            dict(
                amplitude=1.0,
                mean=0.0,
                sigma=1.0,
                period=1.0,
                truncation=1,
            ),
        ),
        (
            integrate_periodic_lorentzian,
            dict(
                amplitude=1.0,
                mean=0.0,
                sigma=1.0,
                period=1.0,
                truncation=1,
            ),
        ),
        (
            integrate_periodic_asymm_gaussian,
            dict(
                amplitude=1.0,
                mean=0.0,
                sigma_1=1.0,
                sigma_2=2.0,
                period=1.0,
                truncation=1,
            ),
        ),
        (
            integrate_periodic_asymm_lorentzian,
            dict(
                amplitude=1.0,
                mean=0.0,
                sigma_1=1.0,
                sigma_2=2.0,
                period=1.0,
                truncation=1,
            ),
        ),
    ]
    for func, pars in func_params:
        assert func(1.0, 0.0, **pars) == 0.0
        assert func(0.5, 0.5, **pars) == 0.0

    def model(x):
        return x

    assert integrate_trapezoid(model, 1.0, 0.0) == 0.0
    assert integrate_trapezoid(model, 0.5, 0.5) == 0.0
