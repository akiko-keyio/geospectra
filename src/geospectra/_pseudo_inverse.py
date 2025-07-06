import warnings
import numpy as np


def pseudo_inverse(
    X: np.ndarray, *, solver: str = "svd", rcond: float = 1e-13
) -> np.ndarray:
    """Compute the Moore-Penrose pseudoinverse.

    Parameters
    ----------
    X : np.ndarray
        Matrix to invert.
    solver : {"normal", "svd"}, default="svd"
        Algorithm used for inversion.
    rcond : float, default=1e-13
        Cutoff for small singular values when using SVD.

    Returns
    -------
    np.ndarray
        Pseudoinverse of ``X``.
    """
    if solver == "normal":
        warnings.warn(
            "`normal` solver 数值不稳定，除教学/调试外请用 `svd`.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.linalg.inv(X.T @ X) @ X.T
    if solver == "svd":
        return np.linalg.pinv(X, rcond=rcond)
    raise ValueError("`solver` must be 'normal' or 'svd'")
