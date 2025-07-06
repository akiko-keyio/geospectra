from .format import CoordsConvert, Polynomial2D, SphericalHarmonics
from .linear_basis_function import Chebyshev2DModel, Polynomial2DModel
from .transform_coding import (
    GeneralizedLinearModel,
    LinearBasis2DTransformer,
    PCA,
)

__all__ = [
    "CoordsConvert",
    "Polynomial2D",
    "SphericalHarmonics",
    "Chebyshev2DModel",
    "Polynomial2DModel",
    "GeneralizedLinearModel",
    "LinearBasis2DTransformer",
    "PCA",
    "main",
]


def main() -> None:
    print("Hello from geospectra!")
