import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

from okkie.astro import B_CONST, DEFAULT_M_NS, DEFAULT_R_NS, Pulsar, PulsarGeom


def test_default_values():
    assert_allclose(DEFAULT_M_NS, 2.7837738e30 * u.kg)
    assert DEFAULT_R_NS == 1.2e6 * u.cm
    assert_allclose(B_CONST, 2.3446677e19 * u.Unit("G s^(-1/2)"))


class TestPulsarBase:
    def setup_class(self):
        self.pulsar = Pulsar(P0=0.1 * u.s, P1=1e-14)

    def test_attributes(self):
        assert_allclose(self.pulsar.R_NS, DEFAULT_R_NS)
        assert_allclose(self.pulsar.M_NS, DEFAULT_M_NS)
        assert_allclose(
            self.pulsar.B_NS,
            (B_CONST * np.sqrt(self.pulsar.P0 * self.pulsar.P1)).to("G"),
        )
        assert_allclose(self.pulsar.F0, 10 / u.s)
        assert_allclose(self.puslar.F1, -1e-12 * u.Unit("s-2"))
        assert_allclose(self.puslar.age, None)
        assert_allclose(self.pulsar.R_LC, 4771345.2 * u.m)
        assert_allclose(self.pulsar.B_LC, 11795.124 * u.G)
        assert_allclose(self.pulsar.I_NS, 1.6034537e45 * u.Unit("cm2 g"))
        assert_allclose(self.pulsar.E_dot, -6.3301816e35 * u.Unit("erg s-1"))
        assert_allclose(self.pulsar.dist, 1 * u.kpc)
        assert_allclose(self.pulsar.Omega, 62.831853 * u.Unit("s-1"))
        assert_allclose(
            self.pulsar.position, SkyCoord(0, 0, frame="galactic", unit="deg")
        )

    def test_geom(self):
        assert isinstance(self.pulsar.geom, PulsarGeom)
        assert self.puslar.geom.alpha == 90 * u.deg
        assert self.pulsar.geom.zeta == 90 * u.deg


def test_from_frequency():
    F0 = 10 / u.s
    F1 = -1e-12 * u.Unit("s-2")
    pulsar = Pulsar.from_frequency(F0=F0, F1=F1)
    assert pulsar.F0 == F0
    assert pulsar.F1 == F1
    assert pulsar.P0 == 0.1 * u.s
    assert pulsar.P1 == 1e-14
