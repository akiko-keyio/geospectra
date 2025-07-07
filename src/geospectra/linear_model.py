"""Lightweight linear regression for precomputed design matrices."""

from __future__ import annotations

from sklearn.base import _fit_context
import numpy as np


import scipy.sparse as sp
from scipy import linalg
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import (
    validate_data,
    _check_sample_weight,
)
from sklearn.linear_model._base import _preprocess_data, _rescale_data


class LinearRegressionCond(LinearRegression):
    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        copy_X: bool = True,
        tol: float = 1e-6,
        n_jobs: int | None = None,
        positive: bool = False,
        cond_threshold: float | str | None = "auto",
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            tol=tol,
            n_jobs=n_jobs,
            positive=positive,
        )
        self.cond_threshold = cond_threshold  # 新增可调超参数

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """
        Identical signature to sklearn's fit; only dense path differs.
        """
        if self.positive or sp.issparse(X):
            return super().fit(X, y, sample_weight=sample_weight)

        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            y_numeric=True,
            multi_output=True,
            force_writeable=True,
        )

        has_sw = sample_weight is not None
        if has_sw:
            sample_weight = _check_sample_weight(
                sample_weight,
                X,
                dtype=X.dtype,
                ensure_non_negative=True,
            )

        copy_X = self.copy_X
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=copy_X,
            sample_weight=sample_weight,
        )

        if has_sw:
            X, y = _rescale_data(X, y, sample_weight, inplace=copy_X)

        # Custom Condition Number Control
        self.cond_threshold_ = self._resolve_cond_threshold(X.shape, X.dtype)
        self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(
            X, y, cond=self.cond_threshold_
        )
        self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = self.coef_.ravel()

        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    # -------------------------- helper -------------------------------
    def _resolve_cond_threshold(self, shape, dtype):
        """
        Translate user `cond` into float|None for numpy.linalg.lstsq.
        See https://www.numberanalytics.com/blog/ultimate-guide-condition-number-determinants
        """

        if self.cond_threshold == "auto":
            return max(shape) * np.finfo(dtype).eps
        if self.cond_threshold is None:
            return None
        if isinstance(self.cond_threshold, (int, float)):
            if self.cond_threshold < 0:
                raise ValueError("`cond` must be non-negative.")
            return float(self.cond_threshold)
        raise ValueError("`cond` must be 'auto', None or a non-negative float.")


__all__ = ["LinearRegressionCond"]
