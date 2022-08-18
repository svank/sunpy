import numpy as np
import pytest

import astropy.coordinates
import astropy.units as u

from sunpy.coordinates.representation import (
    SouthPoleSphericalRepresentation,
    UnitSouthPoleSphericalRepresentation,
)


@pytest.mark.parametrize('unit', [True, False])
def test_SouthPoleSphericalRepresentation_conversions(unit):
    distance = 1*u.AU if not unit else 1
    for lat in np.linspace(0, 180, 10) * u.deg:
        for lon in np.linspace(0, 360, 10, endpoint=False) * u.deg:
            # Test that arguments are correctly stored
            if unit:
                rep = UnitSouthPoleSphericalRepresentation(lon, lat)
            else:
                rep = SouthPoleSphericalRepresentation(lon, lat, distance)

            assert rep.lat == lat
            assert rep.lon == lon
            if not unit:
                assert rep.distance == distance

            # Test the shortcut conversion to SphericalRepresentation
            sp_rep = rep.represent_as(
                    astropy.coordinates.SphericalRepresentation)

            assert sp_rep.lon == lon
            assert u.allclose(sp_rep.lat, lat - 90*u.deg, atol=1e-5*u.deg)
            assert sp_rep.distance == distance

            # Test the shortcut conversion to UnitSphericalRepresentation
            usp_rep = rep.represent_as(
                    astropy.coordinates.UnitSphericalRepresentation)

            assert usp_rep.lon == lon
            assert u.allclose(usp_rep.lat, lat - 90*u.deg, atol=1e-5*u.deg)

            # Test a non-shortcut conversion
            ph_rep = rep.represent_as(
                    astropy.coordinates.PhysicsSphericalRepresentation)

            if lat != 0*u.deg and lat != 180*u.deg:
                # Longitude is arbitrary at the poles
                assert u.allclose(ph_rep.phi, lon, atol=1e-5*u.deg)
            assert u.allclose(ph_rep.theta, 180*u.deg - lat, atol=1e-5*u.deg)

            # Roundtrip that conversion
            rep2 = ph_rep.represent_as(SouthPoleSphericalRepresentation)

            assert u.allclose(rep2.lat, lat, atol=1e-5*u.deg)
            if lat != 0*u.deg and lat != 180*u.deg:
                # Longitude is arbitrary at the poles
                assert u.allclose(rep2.lon, lon, atol=1e-5*u.deg)
            assert u.allclose(rep2.distance, distance)


def test_SouthPoleSphericalRepresentation_boundaries():
    # Test latitude boundaries
    with pytest.raises(ValueError):
        SouthPoleSphericalRepresentation(0*u.deg, -1e-5*u.deg, 1*u.AU)

    with pytest.raises(ValueError):
        SouthPoleSphericalRepresentation(0*u.deg, (180 + 1e-5)*u.deg, 1*u.AU)

    # Shouldn't raise an error
    SouthPoleSphericalRepresentation(0*u.deg, 0*u.deg, 1*u.AU)
    SouthPoleSphericalRepresentation(0*u.deg, 180*u.deg, 1*u.AU)



def test_UnitSouthPoleSphericalRepresentation_boundaries():
    # Test latitude boundaries
    with pytest.raises(ValueError):
        UnitSouthPoleSphericalRepresentation(0*u.deg, -1e-5*u.deg)

    with pytest.raises(ValueError):
        UnitSouthPoleSphericalRepresentation(0*u.deg, (180 + 1e-5)*u.deg)

    # Shouldn't raise an error
    UnitSouthPoleSphericalRepresentation(0*u.deg, 0*u.deg)
    UnitSouthPoleSphericalRepresentation(0*u.deg, 180*u.deg)


def test_SouthPoleSphericalRepresentation_wraparound():
    # Test longitude wraparound
    rep1 = SouthPoleSphericalRepresentation(10*u.deg, 20*u.deg, 1*u.AU)
    rep2 = SouthPoleSphericalRepresentation(370*u.deg, 20*u.deg, 1*u.AU)
    rep3 = SouthPoleSphericalRepresentation(-350*u.deg, 20*u.deg, 1*u.AU)

    assert u.allclose(rep1.lon, rep2.lon)
    assert u.allclose(rep1.lon, rep3.lon)


def test_UnitSouthPoleSphericalRepresentation_wraparound():
    # Test longitude wraparound
    rep1 = UnitSouthPoleSphericalRepresentation(10*u.deg, 20*u.deg)
    rep2 = UnitSouthPoleSphericalRepresentation(370*u.deg, 20*u.deg)
    rep3 = UnitSouthPoleSphericalRepresentation(-350*u.deg, 20*u.deg)

    assert u.allclose(rep1.lon, rep2.lon)
    assert u.allclose(rep1.lon, rep3.lon)

