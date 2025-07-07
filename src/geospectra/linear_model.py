"""Lightweight linear regression for precomputed design matrices."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted


def _pseudo_inverse(
    X: np.ndarray, *, solver: str = "svd", rcond: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return the pseudoinverse of ``X`` along with its singular values and rank."""

    if solver == "normal":
        pinv = np.linalg.inv(X.T @ X) @ X.T
        singular_values = np.linalg.svd(X, compute_uv=False)
        rank = np.linalg.matrix_rank(X)
        return pinv, singular_values, rank

    if solver == "svd":
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        cutoff = S.max() * rcond
        mask = S > cutoff
        Sinv = np.zeros_like(S)
        Sinv[mask] = 1.0 / S[mask]
        pinv = (Vt.T * Sinv) @ U.T
        rank = int(mask.sum())
        return pinv, S, rank

    raise ValueError("solver must be 'normal' or 'svd'")


class BasisFunctionRegressor(RegressorMixin, BaseEstimator):
    """Linear regression estimator working on precomputed features."""

    def __init__(
        self,
        *,
        solver: str = "svd",
        fit_intercept: bool = False,
        rcond: float = 1e-12,
    ) -> None:
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.rcond = rcond

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BasisFunctionRegressor":
        """Fit linear regression on a design matrix ``X``."""

        X = check_array(X, dtype=float, ensure_2d=True)
        y = check_array(y, dtype=float, ensure_2d=False)
        if y.ndim == 1:
            y = y[:, None]

        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X_offset = X.mean(axis=0)
            y_offset = y.mean(axis=0)
            X_centered = X - X_offset
            y_centered = y - y_offset
            self.pinv_matrix_, self.singular_, self.rank_ = _pseudo_inverse(
                X_centered, solver=self.solver, rcond=self.rcond
            )
            coef = self.pinv_matrix_ @ y_centered
            self.coef_ = coef.T
            self.intercept_ = y_offset - X_offset @ self.coef_.T
        else:
            self.pinv_matrix_, self.singular_, self.rank_ = _pseudo_inverse(
                X, solver=self.solver, rcond=self.rcond
            )
            coef = self.pinv_matrix_ @ y
            self.coef_ = coef.T
            self.intercept_ = np.zeros(y.shape[1])

        if self.coef_.shape[0] == 1:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = float(self.intercept_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model."""

        check_is_fitted(self, ["coef_", "pinv_matrix_"])
        X = check_array(X, dtype=float, ensure_2d=True)
        pred = X @ (self.coef_.T)
        return pred + self.intercept_


__all__ = ["BasisFunctionRegressor"]
