import numpy as np
import pandas as pd
from loguru import logger
from geospectra.basis import PolynomialBasis, SphericalHarmonicsBasis


class PCA:
    def __init__(self, criteria="normal"):
        self.mean_ = None
        self.feature_names_in_ = None
        self.n_features = None
        self.components_ = None
        self.components_selected_ = None
        self.explained_variance_ = None
        self.criteria = criteria

    def _validate_and_extract_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Expected input type is pd.DataFrame, but got {}".format(
                    type(X).__name__
                )
            )
        if X.isna().any().any():
            raise ValueError("NaN values in X")

        feature_names = X.columns.tolist()
        n_samples = len(X.index)
        X = X.to_numpy()
        return feature_names, n_samples, X

    def fit(self, X):
        self.feature_names_in_, n_samples, X = self._validate_and_extract_data(X)
        self.n_features = len(self.feature_names_in_)

        # Mean vectors
        self.mean_ = X.mean(axis=0)
        self.mean_ = np.ones_like(self.mean_)
        assert self.mean_.shape == (self.n_features,)

        # Covariance matrix
        X_central = X - self.mean_
        self.Covariance = X_central.T @ X_central
        # self.Covariance /= n_samples - 1

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.Covariance)
        # The column eigenvectors[:, i] is the normalized eigenvector corresponding to the eigenvalue eigenvalues[i].
        # The eigenvalues in ascending order, each repeated according to its multiplicity.
        eigenvalues = np.flip(eigenvalues, axis=0)
        eigenvectors = np.flip(eigenvectors, axis=1)

        # All eigenvalues are positive and in descending order.
        assert (eigenvalues > -1.0e-3).all()
        assert np.all(np.diff(eigenvalues) <= 0)
        eigenvalues[eigenvalues < 0] = 0

        # Principal axes in feature space, representing the directions of maximum variance in the data.
        self.components_ = eigenvectors
        assert self.components_.shape == (self.n_features, self.n_features)

        # The amount of variance explained by each of the selected components.
        self.explained_variance_ = eigenvalues
        return self

    def get_theoretical_rmse(self, total_covariance=None):
        if total_covariance is None:
            total_covariance = np.trace(self.Covariance)

        # Compute the cumulative variance ratio
        cumulative_variance = np.cumsum(self.explained_variance_ * 1.0e10) / 1.0e10
        cumulative_variance_ratio = cumulative_variance / self.explained_variance_.sum()
        cumulative_variance_ratio[cumulative_variance_ratio > 1] = 1

        # Compute the theoretical RMSE
        theoretical_rmse = np.sqrt(
            (1 - cumulative_variance_ratio) * total_covariance / self.n_features
        )
        return theoretical_rmse

    def validate_n_components(self, n_components):
        if n_components < 0 or n_components > self.n_features:
            raise ValueError(
                "n_components must be between 0 and the number of features"
            )

    def select_components(self, n_components=1, X=None):
        if self.criteria == "normal":
            return self.components_[:, :n_components]
        else:
            raise NotImplementedError

    def get_design_matrix(self, n_components, X=None):
        return self.select_components(n_components, X=None).T

    def transform(self, X, n_components):
        # Data validation
        feature_names, _, X = self._validate_and_extract_data(X)
        if feature_names != self.feature_names_in_:
            raise ValueError(
                "The data to transform is not in the same feature space as the principal components"
            )

        # Select partial principal components
        self.components_selected_ = self.select_components(n_components)
        # assert self.components_selected_.shape == (self.n_features, n_components)

        # Project x onto the principal component
        a = (X - self.mean_) @ self.components_selected_

        assert a.shape[1] == n_components

        return a

    def inverse_transform(self, a):
        X_recovered = a @ self.components_selected_.T + self.mean_
        return X_recovered

    def reconstructed(self, X, n_components):
        return self.inverse_transform(self.transform(X, n_components))

    def report(self, n_components):
        return {
            "condition_number": np.linalg.cond(self.components_selected_),
            "param_num": n_components,
        }


class GeneralizedLinearModel:
    def __init__(
        self,
        basis_function,
        variable_names=["lon", "lat"],
        solver="svd",
        fit_intercept=False,
    ):
        """
        Initialize the GeneralizedLinearModel with a specified basis function and solver.

        Parameters:
        - basis_function: A callable that takes lon and lat arrays and returns a transformed feature matrix.
        - solver: The method to use for parameter estimation ('normal' or 'svd').
        """
        self.basis_function = basis_function
        self.variable_names = variable_names
        self.solver = solver
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.intercept_ = None

    def _validate_and_extract_data(self, X):
        """
        Validate the input DataFrame to ensure it contains 'lon' and 'lat' columns.

        Parameters:
        - X: A pandas DataFrame to validate.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Expected input type is pd.DataFrame, but got {}".format(
                    type(X).__name__
                )
            )
        if X.isna().any().any():
            raise ValueError("NaN values in X")

        X = X[self.variable_names].values

        return X

    def fit(self, X, y):
        """
        Fit the model using the provided data.

        Parameters:
        - X: A pandas DataFrame containing 'lon' and 'lat' columns.
        - y: A pandas Series or numpy array containing the target variable.
        """
        X = self._validate_and_extract_data(X)

        # Generate features using the basis function
        X_transformed = self.basis_function.fit_transform(X)
        self.X_transformed = X_transformed

        # Fit intercept_
        y = np.asarray(y)
        self.intercept_ = np.zeros(y.shape[1])
        if self.fit_intercept:
            self.intercept_ = np.mean(y, axis=0)
            y = y - self.intercept_

        # Calculate the pseudo-inverse of the design matrix
        if self.solver == "normal":
            logger.warning("Using normal solver, result may be unreliable")
            X_design_pinv = (
                np.linalg.inv(X_transformed.T @ X_transformed) @ X_transformed.T
            )
        elif self.solver == "svd":
            logger.info("Using svd solver")
            X_design_pinv = np.linalg.pinv(X_transformed, rcond=1e-13)
        else:
            raise ValueError("Solver must be 'normal' or 'svd'.")

        # Compute the coefficients
        self.coef_ = X_design_pinv @ y
        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters:
        - X: A pandas DataFrame containing 'lon' and 'lat' columns.

        Returns:
        - Predictions as a numpy array.
        """
        X = self._validate_and_extract_data(X)

        # Generate features using the basis function
        X_transformed = self.basis_function.transform(X)

        # Predict using the computed coefficients
        return X_transformed @ self.coef_ + self.intercept_[np.newaxis, :]


class LinearBasis2DTransformer:
    """
    This class is designed to transform 2D data using specified basis functions
    (e.g., polynomial, Chebyshev, Legendre, spherical harmonics). It requires
    that the data used in the `transform` method must have the same feature
    columns as those used in the `fit` method. This ensures that the feature
    space remains consistent between fitting and transforming, which is crucial
    for the correct application of the basis transformation and subsequent
    calculations. Any deviation in the feature columns between these methods
    will result in an error, enforcing this consistency.
    """

    def __init__(
        self, basis, variable_names=["lon", "lat"], features_name="site", solver="svd"
    ):
        self.basis = basis
        self.variable_names = variable_names
        self.feature_name = features_name
        self.solver = solver
        if basis == "polynomial_normal_solver":
            self.solver = "normal"

        self.coef_ = None
        self.feature_names_in_ = None

    def _validate_and_extract_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Expected input type is pd.DataFrame, but got {}".format(
                    type(X).__name__
                )
            )
        if X.isna().any().any():
            raise ValueError("NaN values in X")

        feature = X[self.feature_name].to_list()
        X = X[self.variable_names].values

        return X, feature

    def fit(self, X):
        basis_instance_dict = {
            "chebyshev": PolynomialBasis(basis="chebyshev"),
            "polynomial": PolynomialBasis(basis="polynomial"),
            "polynomial_normal_solver": PolynomialBasis(basis="polynomial"),
            "legendre": PolynomialBasis(basis="legendre"),
            "spherical_harmonics": SphericalHarmonicsBasis(cup=False),
            "spherical_cup_harmonics": SphericalHarmonicsBasis(cup=True),
        }

        self.basis_tramsfomer = basis_instance_dict[self.basis]
        self.X, self.feature_names_in_ = self._validate_and_extract_data(X)
        self.basis_tramsfomer.fit(self.X)
        return self

    def transform(self, X):
        if (X.columns != self.feature_names_in_).any():
            raise ValueError(
                "The data to transform is not in the same feature space as the fitted"
            )

        # Transform the input data X into the design matrix using the specified basis transformation
        self.X_design = self.basis_tramsfomer.fit_transform(self.X)

        # Calculate the pseudo-inverse of the design matrix using np.linalg.pinv
        # This method uses Singular Value Decomposition (SVD) to compute the Moore-Penrose pseudo-inverse.
        # Mathematical equal to X_design_pseudo_inverse = np.linalg.inv((X_design.T @ X_design)) @ X_design.T
        # But it is robust and can handle cases where the design matrix is not full rank (i.e., columns are linearly dependent).
        if self.solver == "normal":
            logger.warning("Using normal solver, result may be unreliable")
            self.X_design_pinv = (
                np.linalg.inv((self.X_design.T @ self.X_design)) @ self.X_design.T
            )
        elif self.solver == "svd":
            logger.info("Using svd solver")
            self.X_design_pinv = np.linalg.pinv(self.X_design)
        else:
            raise ValueError()

        return np.asarray(X) @ self.X_design_pinv.T

    def inverse_transform(self, a):
        return a @ self.X_design.T

    def reconstructed(self, X, degree):
        self.basis_tramsfomer.set_params(degree=degree)
        return self.inverse_transform(self.transform(X))

    def report(self, degree):
        X_design = self.basis_tramsfomer.set_params(degree=degree).fit_transform(self.X)
        return {
            "condition_number": np.linalg.cond(X_design),
            "param_num": X_design.shape[1],
        }

    def get_design_matrix(self, degree):
        self.basis_tramsfomer.set_output(transform="pandas")
        self.basis_tramsfomer.set_params(degree=degree)
        return self.basis_tramsfomer.fit_transform(self.X).T
