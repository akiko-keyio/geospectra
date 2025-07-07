import numpy as np
import pytest

from geospectra import SphericalHarmonicsBasis, BasisFunctionRegressor


def generate_data(
    degree: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate design matrix and target for a given spherical harmonic degree."""
    n_features = (degree + 1) * (degree + 2) // 2
    lon = rng.uniform(-np.pi, np.pi, size=n_features)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=n_features)
    X = np.column_stack([lon, lat])
    basis = SphericalHarmonicsBasis(
        degree=degree,
        cup=True,
        include_bias=True,
        force_norm=True,
    )
    design = basis.fit_transform(X)
    coef = rng.normal(size=n_features)
    y = design @ coef
    return design, y, coef


@pytest.mark.parametrize("degree", [1, 5, 10])
def test_spherical_regressor_stability(degree: int) -> None:
    rng = np.random.default_rng(degree)
    design, y, coef = generate_data(degree, rng)
    reg = BasisFunctionRegressor(fit_intercept=False)
    reg.fit(design, y)
    y_pred = reg.predict(design)
    assert np.allclose(y_pred, y, atol=1e-10)
    assert np.allclose(reg.coef_, coef, atol=1e-10)
