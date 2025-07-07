import numpy as np
import pytest

from geospectra import SphericalHarmonicsBasis


def generate_data(
    degree: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate design matrix and target for a given spherical harmonic degree."""
    basis = SphericalHarmonicsBasis(
        degree=degree,
        cup=False,
        include_bias=False,
    )

    lon = rng.uniform(-180, 180, size=(degree + 1) ** 2 * 2)
    lat = rng.uniform(-90, 90, size=(degree + 1) ** 2 * 2)
    X = np.column_stack([lon, lat])

    design = basis.fit_transform(X)

    coef = rng.normal(size=basis.n_output_features_)
    intercept = rng.normal()
    y = design @ coef + intercept

    print(f"n={basis.n_output_features_} r={X.shape[0]}")

    return design, y, coef, intercept


@pytest.mark.parametrize("degree", [5, 10, 25, 40])
def test_spherical_regressor_stability(degree: int) -> None:
    rng = np.random.default_rng(degree)
    design, y, coef, intercept = generate_data(degree, rng)
    from geospectra.linear_model import LinearRegressionCond

    reg = LinearRegressionCond()
    reg.fit(design, y)

    dcoef = np.linalg.norm(reg.coef_ - coef)
    dintercept = reg.intercept_ - intercept

    y_pred = reg.predict(design)
    dy = np.linalg.norm(y_pred - y)

    assert dy < 1e-3
    assert dcoef < 1e-3
    assert dintercept < 1e-3
