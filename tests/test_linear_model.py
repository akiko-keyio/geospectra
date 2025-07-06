import numpy as np
import pytest

from geospectra.basis import PolynomialBasis, SphericalHarmonicsBasis
from geospectra.linear_model import BasisFunctionRegressor
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


def _make_polynomial_data() -> tuple[np.ndarray, np.ndarray, PolynomialBasis]:
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(20, 2))
    basis = PolynomialBasis(degree=2, include_bias=True)
    design = basis.fit_transform(X)
    coef = rng.normal(size=(design.shape[1], 2))
    intercept = rng.normal(size=2)
    y = design @ coef + intercept
    return X, y, basis


def _make_spherical_data() -> tuple[np.ndarray, np.ndarray, SphericalHarmonicsBasis]:
    rng = np.random.default_rng(1)
    lon = rng.uniform(-np.pi, np.pi, size=15)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=15)
    X = np.column_stack([lon, lat])
    basis = SphericalHarmonicsBasis(degree=2, cup=False, include_bias=True)
    design = basis.fit_transform(X)
    coef = rng.normal(size=(design.shape[1], 1))
    intercept = rng.normal(size=1)
    y = design @ coef + intercept
    return X, y, basis


def test_polynomial_fit_predict_multi_target() -> None:
    X, y, basis = _make_polynomial_data()
    reg = BasisFunctionRegressor(basis=basis, fit_intercept=True)
    reg.fit(X, y)
    assert reg.coef_.shape == (basis.n_output_features_, 2)
    pred = reg.predict(X)
    assert np.allclose(pred, y)


def test_spherical_fit_predict_single_target() -> None:
    X, y, basis = _make_spherical_data()
    reg = BasisFunctionRegressor(basis=basis, fit_intercept=True)
    reg.fit(X, y)
    pred = reg.predict(X)
    assert np.allclose(pred, y)


def test_design_matrix_reuse() -> None:
    X, y, basis = _make_polynomial_data()
    reg = BasisFunctionRegressor(basis=basis, fit_intercept=True)
    reg.fit(X, y)
    design_id = id(reg.design_matrix_)
    _ = reg.predict(X)
    assert id(reg.design_matrix_) == design_id


def test_incompatible_feature_space() -> None:
    X, y, basis = _make_polynomial_data()
    reg = BasisFunctionRegressor(basis=basis, fit_intercept=True)
    reg.fit(X, y)
    with pytest.raises(ValueError):
        reg.predict(np.ones((3, 3)))


def test_pipeline_compatibility() -> None:
    X, y, basis = _make_polynomial_data()
    pipe = make_pipeline(BasisFunctionRegressor(basis=basis, fit_intercept=True))
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert np.allclose(pred, y)


def test_grid_search_cv() -> None:
    X, y, _ = _make_polynomial_data()
    grid = {
        "basis__degree": [1, 2],
        "fit_intercept": [True, False],
    }
    gs = GridSearchCV(BasisFunctionRegressor(basis=PolynomialBasis()), grid, cv=2)
    gs.fit(X, y)
    assert isinstance(gs.best_estimator_, BasisFunctionRegressor)


def test_clone_estimator() -> None:
    reg = BasisFunctionRegressor(basis=PolynomialBasis(degree=3))
    reg2 = clone(reg)
    assert reg is not reg2
    assert reg2.basis.degree == 3
