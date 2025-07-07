from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ._pseudo_inverse import pseudo_inverse


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
    rcond : float, default=1e-13
        Cutoff for small singular values when using SVD.
    """

    def __init__(
        self,
        *,
        basis: Optional[object] = None,
        solver: str = "svd",
        fit_intercept: bool = False,
        rcond: float = 1e-13,
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
        y = check_array(y, dtype=float, ensure_2d=True)

        self.X_ = X
        self.design_matrix_ = self.basis.fit_transform(X)
        self.pinv_matrix_ = pseudo_inverse(
            self.design_matrix_, solver=self.solver, rcond=self.rcond
        )

        if self.fit_intercept:
            self.intercept_ = y.mean(axis=0)
            y = y - self.intercept_
        else:
            self.intercept_ = np.zeros(y.shape[1])

        self.coef_ = self.pinv_matrix_ @ y
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
                raise ValueError("新数据的特征空间与训练数据不一致，无法进行预测。")

        return X_design @ self.coef_ + self.intercept_


__all__ = ["BasisFunctionRegressor"]
