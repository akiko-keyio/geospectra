"""Check regression stability for spherical harmonics basis."""

from __future__ import annotations

import numpy as np

from geospectra import BasisFunctionRegressor, SphericalHarmonicsBasis


def evaluate_degree(degree: int, rng: np.random.Generator) -> tuple[float, float]:
    """Return residual and coefficient error for a given degree."""
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
    coef_true = rng.normal(size=n_features)
    y = design @ coef_true
    reg = BasisFunctionRegressor(fit_intercept=False)
    reg.fit(design, y)
    residual = np.linalg.norm(reg.predict(design) - y)
    coef_err = np.linalg.norm(reg.coef_ - coef_true)
    return residual, coef_err


def main() -> None:
    rng = np.random.default_rng(0)
    for degree in range(1, 51):
        residual, coef_err = evaluate_degree(degree, rng)
        print(f"degree={degree:2d} residual={residual:.2e} coef_error={coef_err:.2e}")


if __name__ == "__main__":
    main()
