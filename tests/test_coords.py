import numpy as np
from geospectra.basis import CoordsConverter


def test_transform_handles_pole_point() -> None:
    lon = np.array([10.0, 20.0, 30.0])
    lat = np.array([40.0, 50.0, 60.0])
    conv = CoordsConverter(pole="xyzmean", method="central")
    conv.fit(lon, lat)
    lon0 = np.degrees(conv.phi0)
    lat0 = 90 - np.degrees(conv.theta0)
    theta, phi = conv.transform(np.array([lon0]), np.array([lat0]))
    assert not np.isnan(phi).any()
    assert np.allclose(phi, 0.0)
