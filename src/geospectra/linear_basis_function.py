"""Simple linear models built on basis transformers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .basis import PolynomialBasis


class LinearBasisModel(BaseEstimator, RegressorMixin):
    """Linear model using a basis transformer."""

    def __init__(self, transformer: PolynomialBasis) -> None:
        self.transformer = transformer
        self.coef_: np.ndarray | None = None
        self.degree = 0

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, degree: int = 1
    ) -> "LinearBasisModel":
        X, y = self._validate_data(X, y, multi_output=True, y_numeric=True)
        self.degree = degree
        self.transformer.set_params(degree=degree)
        self.transformer.fit(X)
        design = self.transformer.transform(X)
        self._pinv = np.linalg.pinv(design)
        if y is not None:
            self.coef_ = self._pinv @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_")
        design = self.transformer.transform(X)
        return design @ self.coef_


class PolynomialModel(LinearBasisModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(PolynomialBasis(basis="polynomial", **kwargs))


class ChebyshevModel(LinearBasisModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(PolynomialBasis(basis="chebyshev", **kwargs))
