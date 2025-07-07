import numpy as np
import pandas as pd


class PCA:
    def __init__(self, *, criteria="normal") -> None:
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
