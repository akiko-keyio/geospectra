"""Simple linear models built on basis transformers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .basis import PolynomialBasis


class LinearBasisModel:
    """Linear model using a basis transformer."""

    def __init__(self, transformer: PolynomialBasis) -> None:
        self.transformer = transformer
        self.coef_: np.ndarray | None = None
        self.degree = 0

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None, degree: int = 1
    ) -> "LinearBasisModel":
        self.degree = degree
        self.transformer.set_params(degree=degree)
        self.transformer.fit(X)
        design = self.transformer.transform(X)
        self._pinv = np.linalg.pinv(design)
        if y is not None:
            self.coef_ = self._pinv @ np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        design = self.transformer.transform(X)
        if self.coef_ is None:
            raise ValueError("Model is not fitted")
        return design @ self.coef_


class PolynomialModel(LinearBasisModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(PolynomialBasis(basis="polynomial", **kwargs))


class ChebyshevModel(LinearBasisModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(PolynomialBasis(basis="chebyshev", **kwargs))
