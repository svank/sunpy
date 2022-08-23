import numpy as np
import pytest

import astropy.coordinates
import astropy.units as u

from sunpy.coordinates.representation import (
    SouthPoleSphericalRepresentation,
    UnitSouthPoleSphericalRepresentation,
    SouthPoleSphericalDifferential,
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


def test_SouthPoleSphericalDifferential_roundtrip():
    # The only difference between SphericalRepresentation and
    # SouthPoleSphericalRepresentation is a constant offset in latitude, so
    # SphericalDifferential and SouthPoleSphericalDifferential should be the
    # same. Let's make a bunch of each, transform them to the other, and check
    # that.

    r = 4 * u.m
    for lat in np.linspace(0, 350, 3) * u.deg:
        # Avoid being right at the poles, where latitude becomes degenerate
        for lon in np.linspace(-89, 89, 3) * u.deg:
            for dlat in np.linspace(-10, 10, 3) * u.deg / u.s:
                for dlon in np.linspace(-10, 10, 3) * u.deg / u.s:
                    for dr in np.linspace(-2, 2, 3) * u.m / u.s:
                        sd = astropy.coordinates.SphericalDifferential(
                                dlat, dlon, dr)

                        base = astropy.coordinates.SphericalRepresentation(
                                lat, lon, r)

                        spsd = sd.represent_as(
                                SouthPoleSphericalDifferential, base=base)

                        assert u.allclose(
                                sd.d_lat, spsd.d_lat, atol=1e-13*u.deg/u.s)
                        assert u.allclose(
                                sd.d_lon, spsd.d_lon, atol=1e-13*u.deg/u.s)
                        assert u.allclose(
                                sd.d_distance, spsd.d_distance,
                                atol=1e-13*u.m/u.s)

                        spsd = SouthPoleSphericalDifferential(
                                dlat, dlon, dr)

                        base = SouthPoleSphericalRepresentation(
                                lat, lon + 90 * u.deg, r)

                        sd = spsd.represent_as(
                                astropy.coordinates.SphericalDifferential,
                                base=base)

                        assert u.allclose(
                                sd.d_lat, spsd.d_lat, atol=1e-13*u.deg/u.s)
                        assert u.allclose(
                                sd.d_lon, spsd.d_lon, atol=1e-13*u.deg/u.s)
                        assert u.allclose(
                                sd.d_distance, spsd.d_distance,
                                atol=1e-13*u.m/u.s)


def test_SouthPoleSphericalDifferential_application():
    # Let's create a bunch of SouthPoleSphericalRepresentations and
    # Differentials and equivalent SphericalRepresentations and Differentials.
    # Then let's apply each Differential to its Representation and make sure we
    # get the same Cartesian coordinates from both.

    r = 4 * u.m
    for lat in np.linspace(0, 350, 3) * u.deg:
        # Avoid being right at the poles, where latitude becomes degenerate
        for lon in np.linspace(-89, 89, 3) * u.deg:
            for dlat in np.linspace(-10, 10, 3) * u.deg / u.s:
                for dlon in np.linspace(-10, 10, 3) * u.deg / u.s:
                    for dr in np.linspace(-2, 2, 3) * u.m / u.s:
                        s_diff = astropy.coordinates.SphericalDifferential(
                                dlat, dlon, dr)

                        s_base = astropy.coordinates.SphericalRepresentation(
                                lat, lon, r)

                        s_result = (s_base + s_diff * u.s).represent_as(
                                astropy.coordinates.CartesianRepresentation)

                        sp_diff = SouthPoleSphericalDifferential(
                                dlat, dlon, dr)

                        sp_base = SouthPoleSphericalRepresentation(
                                lat, lon + 90*u.deg, r)

                        sp_result = (sp_base + sp_diff * u.s).represent_as(
                                astropy.coordinates.CartesianRepresentation)

                        assert u.allclose(
                                s_result.x, sp_result.x,
                                atol=1e-13*u.m)
                        assert u.allclose(
                                s_result.y, sp_result.y,
                                atol=1e-13*u.m)
                        assert u.allclose(
                                s_result.z, sp_result.z,
                                atol=1e-13*u.m)

