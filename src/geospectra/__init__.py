from .basis import CoordsConverter, PolynomialBasis, SphericalHarmonicsBasis
from .linear_model import BasisFunctionRegressor
from .transform_coding import (
    LinearBasis2DTransformer,
    PCA,
)

__all__ = [
    "CoordsConverter",
    "PolynomialBasis",
    "SphericalHarmonicsBasis",
    "BasisFunctionRegressor",
    "LinearBasis2DTransformer",
    "PCA",
    "main",
]


def main() -> None:
    print("Hello from geospectra!")
