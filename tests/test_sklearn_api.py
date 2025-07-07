import numpy as np
import pandas as pd
import pytest

from geospectra import SphericalHarmonicsBasis


def test_transform_checks_feature_count() -> None:
    rng = np.random.default_rng(0)
    X = rng.random((5, 2))
    basis = SphericalHarmonicsBasis()
    basis.fit(X)
    with pytest.raises(ValueError):
        basis.transform(rng.random((2, 3)))


def test_transform_checks_feature_names() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((5, 2)), columns=["lon", "lat"])
    basis = SphericalHarmonicsBasis()
    basis.fit(X)
    X_bad = pd.DataFrame(rng.random((2, 2)), columns=["x", "y"])
    with pytest.raises(ValueError):
        basis.transform(X_bad)


def test_get_feature_names_out_validates_features() -> None:
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.random((5, 2)), columns=["lon", "lat"])
    basis = SphericalHarmonicsBasis()
    basis.fit(X)
    with pytest.raises(ValueError):
        basis.get_feature_names_out(["x", "y"])
