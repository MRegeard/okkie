import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

from okkie.astro import (
    B_CONST,
    DEFAULT_M_NS,
    DEFAULT_R_NS,
    GJ_density,
    Pulsar,
    PulsarGeom,
    rlc,
)
from okkie.data.test import get_test_filepath


def test_default_values():
    assert_allclose(DEFAULT_M_NS, 2.7837738e30 * u.kg)
    assert DEFAULT_R_NS == 1.2e6 * u.cm
    assert_allclose(B_CONST, 2.3446677e19 * u.Unit("G s^(-1/2)"))


class TestPulsarBase:
    def setup_class(self):
        self.pulsar = Pulsar(P0=0.1 * u.s, P1=1e-14)

    def test_attributes(self):
        assert_allclose(self.pulsar.P0, 0.1 * u.s)
        assert_allclose(self.pulsar.P1, 1e-14)
        assert_allclose(self.pulsar.R_NS, DEFAULT_R_NS)
        assert_allclose(self.pulsar.M_NS, DEFAULT_M_NS)
        assert_allclose(
            self.pulsar.B_NS,
            (B_CONST * np.sqrt(self.pulsar.P0 * self.pulsar.P1)).to("G"),
        )
        assert_allclose(self.pulsar.F0, 10 / u.s)
        assert_allclose(self.pulsar.F1, -1e-12 * u.Unit("s-2"))
        assert self.pulsar.age is None
        assert_allclose(self.pulsar.R_LC, 4771345.2 * u.m)
        assert_allclose(self.pulsar.B_LC, 11795.124 * u.G)
        assert_allclose(self.pulsar.I_NS, 1.6034537e45 * u.Unit("cm2 g"))
        assert_allclose(self.pulsar.E_dot, -6.3301816e35 * u.Unit("erg s-1"))
        assert_allclose(self.pulsar.dist, 1 * u.kpc)
        assert_allclose(self.pulsar.Omega, 62.831853 * u.Unit("s-1"))
        assert self.pulsar.position == SkyCoord(0, 0, frame="galactic", unit="deg")

    def test_geom(self):
        assert isinstance(self.pulsar.geom, PulsarGeom)
        assert self.pulsar.geom.alpha == 90 * u.deg
        assert self.pulsar.geom.zeta == 90 * u.deg

    def test_B(self):
        assert_allclose(self.pulsar.B(0.1 * rlc), 11795124 * u.G)
        assert_allclose(self.pulsar.B(100 * u.km), 1.2812239e9 * u.G)

        with pytest.raises(AttributeError):
            self.pulsar.B(30)


def test_pulsar_init():
    pulsar = Pulsar(
        P0=0.1 * u.s,
        P1=1e-14,
        B_NS=10000 * u.G,
        R_NS=10 * u.km,
        age=10000 * u.yr,
        geom=PulsarGeom(70 * u.deg, 45 * u.deg),
        M_NS=1e30 * u.kg,
        dist=3 * u.kpc,
        name="Pulsar test",
        position=SkyCoord(45, 3, frame="icrs", unit="deg"),
    )

    assert_allclose(pulsar.P0, 0.1 * u.s)
    assert_allclose(pulsar.P1, 1e-14)
    assert_allclose(pulsar.B_NS, 10000 * u.G)
    assert_allclose(pulsar.R_NS, 1e6 * u.cm)
    assert_allclose(pulsar.age, 10000 * u.yr)
    assert pulsar.geom.alpha == 70 * u.deg
    assert pulsar.geom.zeta == 45 * u.deg
    assert pulsar.position == SkyCoord(45, 3, frame="icrs", unit="deg")
    assert_allclose(pulsar.dist, 3 * u.kpc)
    assert pulsar.name == "Pulsar test"


def test_pulsar_from_frequency():
    F0 = 10 / u.s
    F1 = -1e-12 * u.Unit("s-2")
    pulsar = Pulsar.from_frequency(F0=F0, F1=F1)
    assert pulsar.F0 == F0
    assert_allclose(pulsar.F1, F1)
    assert pulsar.P0 == 0.1 * u.s
    assert pulsar.P1 == 1e-14


def test_pulsar_from_ephemeris():
    filepath = get_test_filepath("J0633+0632_LAT.par")
    pulsar = Pulsar.from_ephemeris_file(filepath, name="PSR J0614-3329")
    assert_allclose(pulsar.P0, 0.29740521 * u.s)
    assert_allclose(pulsar.P1, 7.957431e-14)
    assert pulsar.name == "PSR J0614-3329"


def test_GJ_density():
    assert_allclose(
        GJ_density(62 / u.s, 0.1 * rlc, 1e7 * u.G, 90 * u.deg, 90 * u.deg),
        -6.790724e-29 * u.Unit("C cm-3"),
    )
    assert_allclose(
        GJ_density(62 / u.s, 0.1 * rlc, 1e7 * u.G, 0 * u.deg, 90 * u.deg),
        -1.10900938e-12 * u.Unit("C / cm3"),
    )
    assert_allclose(
        GJ_density(62 / u.s, 0.9 * rlc, 1e7 * u.G, 90 * u.deg, 90 * u.deg),
        -3.5383245e-28 * u.Unit("C / cm3"),
    )


class TestPulsarGeomBase:
    def setup_class(self):
        self.geom = PulsarGeom()

    def test_attributes(self):
        assert self.geom.alpha == 90 * u.deg
        assert self.geom.zeta == 90 * u.deg

    def reassign_attributes(self):
        self.geom.alpha = 45 * u.deg
        self.geom.zeta = 45 * u.deg
        assert self.geom.alpha == 45 * u.deg
        assert self.geom.zeta == 45 * u.deg
