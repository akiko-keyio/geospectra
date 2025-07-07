from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
import warnings


def _pseudo_inverse(
    X: np.ndarray, *, solver: str = "svd", rcond: float = 2**-52
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return the pseudoinverse of ``X`` along with its singular values and rank."""
    if solver == "normal":
        warnings.warn(
            "'normal' solver is numerically unstable; prefer 'svd'.",
            RuntimeWarning,
            stacklevel=2,
        )
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
    """Regressor using a basis function transformer.

    Parameters
    ----------
    basis : object, default=None
        Transformer with ``fit_transform`` and ``transform`` methods.
    solver : {"normal", "svd"}, default="svd"
        Solver to compute the pseudoinverse.
    fit_intercept : bool, default=False
        Whether to estimate an intercept.
    rcond : float, default=2**-52
        Cutoff for small singular values when using SVD.
    """

    def __init__(
        self,
        *,
        basis: Optional[object] = None,
        solver: str = "svd",
        fit_intercept: bool = False,
        rcond: float = 2**-52,
    ) -> None:
        self.basis = basis
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.rcond = rcond

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BasisFunctionRegressor":
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_targets)
            Target values.
        """
        X = check_array(X, dtype=float, ensure_2d=True)
        y = check_array(y, dtype=float, ensure_2d=False)
        if y.ndim == 1:
            y = y[:, None]

        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.design_matrix_ = self.basis.fit_transform(X)

        if self.fit_intercept:
            X_offset = self.design_matrix_.mean(axis=0)
            y_offset = y.mean(axis=0)
            X_centered = self.design_matrix_ - X_offset
            y_centered = y - y_offset
            self.pinv_matrix_, self.singular_, self.rank_ = _pseudo_inverse(
                X_centered, solver=self.solver, rcond=self.rcond
            )
            coef = self.pinv_matrix_ @ y_centered
            self.coef_ = coef.T
            self.intercept_ = y_offset - X_offset @ self.coef_.T
        else:
            self.pinv_matrix_, self.singular_, self.rank_ = _pseudo_inverse(
                self.design_matrix_, solver=self.solver, rcond=self.rcond
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
        if np.array_equal(X, getattr(self, "X_", None)):
            X_design = self.design_matrix_
        else:
            X_design = self.basis.transform(X)
            if X_design.shape[1] != self.design_matrix_.shape[1]:
                raise ValueError(
                    "Feature space of new data does not match training data."
                )

        pred = X_design @ (self.coef_.T)
        return pred + self.intercept_


__all__ = ["BasisFunctionRegressor"]
