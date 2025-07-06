from abc import abstractmethod
import numpy as np
from numpy.polynomial import Chebyshev, Legendre


class DesignMatrixTransformer:
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X, degree):
        pass

    def fit_transform(self, X, degree):
        return self.fit(X).transform(X, degree)


class Polynomial2D(DesignMatrixTransformer):
    def __init__(self):
        self.degree = None
        self.n_terms = None
        self.terms = None

    def fit(self, X):
        X = np.asarray(X)
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        return self

    def transform(self, X, degree):
        self.degree = degree
        self.n_terms = (degree + 1) * (degree + 2) // 2
        self.terms = []
        X = np.asarray(X)
        X = 2 * (X - self.min_vals) / (self.max_vals - self.min_vals) - 1
        X1, X2 = X[:, 0], X[:, 1]
        X_design = np.ones((len(X), self.n_terms))
        index = 0
        self.terms = []
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                if i == 0 and j == 0:
                    X_design[:, index] = 1
                    self.terms.append("1")
                    index += 1
                    continue
                X_design[:, index] = (X1**i) * (X2**j)
                X1_str = f"X1^{i}" if i != 0 else ""
                X2_str = f"X2^{j}" if j != 0 else ""
                self.terms.append(X1_str + X2_str)
                index += 1
        assert index == self.n_terms
        return X_design


class Legendre2D(DesignMatrixTransformer):
    def __init__(self):
        self.degree = None
        self.n_terms = None
        self.terms = None

    def fit(self, X):
        X = np.asarray(X)
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        return self

    def transform(self, X, degree):
        self.degree = degree
        self.n_terms = (degree + 1) * (degree + 2) // 2
        self.terms = []
        X = np.asarray(X)
        X = 2 * (X - self.min_vals) / (self.max_vals - self.min_vals) - 1
        X1, X2 = X[:, 0], X[:, 1]
        X_design = np.ones((len(X), self.n_terms))
        index = 0
        self.terms = []
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                X_design[:, index] = Legendre.basis(i)(X1) * Legendre.basis(j)(X2)
                self.terms.append(f"Legendre_{i}_{j}")
                index += 1
        assert index == self.n_terms
        return X_design


class Chebyshev2D(DesignMatrixTransformer):
    def __init__(self):
        self.degree = None
        self.n_terms = None
        self.terms = None

    def fit(self, X):
        X = np.asarray(X)
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        return self

    def transform(self, X, degree):
        self.degree = degree
        self.n_terms = (degree + 1) * (degree + 2) // 2
        self.terms = []
        X = np.asarray(X)
        X = 2 * (X - self.min_vals) / (self.max_vals - self.min_vals) - 1
        X1, X2 = X[:, 0], X[:, 1]
        X_design = np.ones((len(X), self.n_terms))
        index = 0
        self.terms = []
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                if i == 0 and j == 0:
                    X_design[:, index] = 1
                    self.terms.append("1")
                else:
                    T_i = Chebyshev.basis(i)
                    T_j = Chebyshev.basis(j)
                    X_design[:, index] = T_i(X1) * T_j(X2)
                    X1_str = f"T_{i}(X1)" if i != 0 else ""
                    X2_str = f"T_{j}(X2)" if j != 0 else ""
                    self.terms.append(X1_str + X2_str)
                index += 1
        assert index == self.n_terms
        return X_design


class CoordsConvert:
    def __init__(self, pole="haversine", method="central_scale", hemisphere_scale=1.0):
        """
        Initialize the CoordsConvert class.

        :param pole: Tuple of (latitude, longitude) or 'calculate' to determine the pole automatically.
        :param method: The method to use for conversion ('non', 'basic', 'central', 'central_scale').
        :param hemisphere_scale: Scale factor for the hemisphere.
        """
        self.pole = pole
        self.method = method
        self.hemisphere_scale = hemisphere_scale
        self.scale = None
        self.theta0 = None
        self.phi0 = None

    def fit(self, lon, lat):
        if self.pole == "haversine":
            lon0, lat0 = self._calculate_central_haversine(lon, lat)
            self.theta0 = np.radians(90 - lat0) + 1.0e-6
            self.phi0 = np.radians(lon0)
        elif self.pole == "xyzmean":
            lon0, lat0 = self._calculate_central_xyzmean(lon, lat)
            self.theta0 = np.radians(90 - lat0)
            self.phi0 = np.radians(lon0)
        elif (
            isinstance(self.pole, tuple)
            and len(self.pole) == 2
            and all(isinstance(item, float) for item in self.pole)
        ):
            self.theta0 = np.radians(90 - self.pole[0])
            self.phi0 = np.radians(self.pole[1])
        else:
            raise ValueError(
                "Pole must be a tuple of two floats or ['haversine','xyzmean']."
            )
        return self

    def transform(self, lon, lat):
        if self.theta0 is None or self.phi0 is None:
            raise AttributeError(
                "theta0 or phi0 is None, please call the fit method first"
            )
        if self.method == "non":
            return self._non(lon, lat)
        elif self.method == "basic":
            return self._basic(lon, lat)
        elif self.method == "central":
            return self._central(lon, lat)
        elif self.method == "central_scale":
            return self._central_scale(lon, lat)
        else:
            raise ValueError(
                'Method must be one of "non", "basic", "central", or "central_scale".'
            )

    def fit_transform(self, lon, lat):
        """
        Convert the given longitude and latitude using the specified method.

        :param lon: Array of longitudes.
        :param lat: Array of latitudes.
        :return: Converted coordinates.
        """
        self.fit(lon, lat)
        return self.transform(lon, lat)

    def plot_convert(self, lon, lat, color=None, cmid=0, **kwargs):
        """
        Plot the converted coordinates using Plotly.

        :param lon: Array of longitudes.
        :param lat: Array of latitudes.
        :param color: Color for the plot.
        :param cmid: Center of the color scale.
        :param kwargs: Additional keyword arguments for layout.
        :return: Plotly Figure object.
        """
        import plotly.graph_objs as go
        import plotly.express as px

        phi, theta = self.fit_transform(lon, lat)
        phi = np.degrees(phi)
        theta = np.degrees(theta)
        return go.Figure(
            go.Scatterpolar(
                r=phi,
                theta=theta,
                mode="markers",
                hovertext=color,
                marker=dict(
                    size=4,
                    color=color,
                    cmid=cmid,
                    colorscale=px.colors.sequential.RdBu,
                    showscale=True,
                ),
            )
        ).update_layout(**kwargs)

    def _calculate_central_haversine(self, lon, lat):
        """
        Calculate the central point using the haversine formula.

        :param lon: Array of longitudes.
        :param lat: Array of latitudes.
        :return: Tuple of (central longitude, central latitude).
        """

        def haversine_np(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            )
            c = 2 * np.arcsin(np.sqrt(a))
            return c

        dist_matrix = haversine_np(lon[:, None], lat[:, None], lon, lat)
        max_dists = dist_matrix.max(axis=1)
        centroid_idx = np.argmin(max_dists)
        return lon[centroid_idx], lat[centroid_idx]

    def _calculate_central_xyzmean(self, lon, lat):
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)

        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_mean = np.mean(z)

        central_lon = np.arctan2(y_mean, x_mean)
        central_lat = np.arctan2(z_mean, np.sqrt(x_mean**2 + y_mean**2))

        central_lon = np.degrees(central_lon)
        central_lat = np.degrees(central_lat)

        return central_lon, central_lat

    def _non(self, lon, lat):
        theta = np.radians(lon)  # Longitude
        phi = np.radians(lat)  # Latitude to colatitude
        return theta, phi

    def _basic(self, lon, lat):
        theta = np.radians(lon)  # Longitude
        phi = np.radians(90 - lat)  # Latitude to colatitude
        return theta, phi

    def _central(self, lon, lat):
        # Reference：
        # GNSS_MODIS_ER...据融合的球冠谐水汽模型构建_樊鉴庆.pdf p55
        # 球冠谐方法精化区域大地水准面_储王宁.pdf p46
        # 基于球冠谐分析的区域精密对流层建模
        theta = np.radians(90 - lat)  # Colatitude
        phi = np.radians(lon)  # Longitude

        # Convert to colatitude with the pole as the center
        theta1 = np.arccos(
            np.cos(self.theta0) * np.cos(theta)
            + np.sin(self.theta0) * np.sin(theta) * np.cos(self.phi0 - phi)
        )

        # Convert to longitude with the pole as the center
        sin_phi1 = np.sin(theta) * np.sin(phi - self.phi0) / np.sin(theta1)
        cos_phi1 = (
            np.sin(self.theta0) * np.cos(theta)
            - np.cos(self.theta0) * np.sin(theta) * np.cos(phi - self.phi0)
        ) / np.sin(theta1)
        phi1 = np.arctan2(sin_phi1, cos_phi1)

        return theta1, phi1

    def _central_scale(self, lon, lat):
        theta1, phi1 = self._central(lon, lat)
        if self.scale is None:
            self.scale = np.pi * self.hemisphere_scale / theta1.max()
        theta_scale = theta1 * self.scale
        return theta_scale, phi1


class LinearBasisModel:
    def __init__(self, design_matrix_builder):
        self.design_matrix_builder = design_matrix_builder
        self.coef_ = None

    def fit(self, X, y=None, degree=1):
        self.degree = degree
        # Transform the input data X into the design matrix using the specified transformation
        self.X_design = self.design_matrix_builder.fit_transform(X, degree)

        # Calculate the pseudo-inverse of the design matrix using np.linalg.pinv
        # This method uses Singular Value Decomposition (SVD) to compute the Moore-Penrose pseudo-inverse.
        # Mathematical equal to X_design_pseudo_inverse = np.linalg.inv((X_design.T @ X_design)) @ X_design.T
        # But it is robust and can handle cases where the design matrix is not full rank (i.e., columns are linearly dependent).
        self.X_design_pseudo_inverse = np.linalg.pinv(self.X_design)

        if y is not None:
            self.coef_ = self.X_design_pseudo_inverse @ np.asarray(
                y
            )  # self.coef_, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        return self

    def predict(self, X):
        X_design = self.design_matrix_builder.transform(X, self.degree)
        return X_design @ self.coef_


# class SphericalHarmonicsModel(LinearBasisModel):
#     def __init__(self, **kwargs):
#         super().__init__(design_matrix_builder=SphericalHarmonics(**kwargs))


class Polynomial2DModel(LinearBasisModel):
    def __init__(self, **kwargs):
        super().__init__(design_matrix_builder=Polynomial2D(**kwargs))


class Chebyshev2DModel(LinearBasisModel):
    def __init__(self, **kwargs):
        super().__init__(design_matrix_builder=Chebyshev2D(**kwargs))
