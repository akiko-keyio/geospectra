from .basis import CoordsConverter, PolynomialBasis, SphericalHarmonicsBasis
from .linear_basis_function import ChebyshevModel, PolynomialModel
from .transform_coding import (
    GeneralizedLinearModel,
    LinearBasis2DTransformer,
    PCA,
)

__all__ = [
    "CoordsConverter",
    "PolynomialBasis",
    "SphericalHarmonicsBasis",
    "ChebyshevModel",
    "PolynomialModel",
    "GeneralizedLinearModel",
    "LinearBasis2DTransformer",
    "PCA",
    "main",
]


def main() -> None:
    print("Hello from geospectra!")
