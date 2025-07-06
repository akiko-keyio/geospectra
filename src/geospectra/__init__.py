from .basis import CoordsConverter, PolynomialBasis, SphericalHarmonicsBasis
from .transform_coding import (
    GeneralizedLinearModel,
    LinearBasis2DTransformer,
    PCA,
)

__all__ = [
    "CoordsConverter",
    "PolynomialBasis",
    "SphericalHarmonicsBasis",
    "GeneralizedLinearModel",
    "LinearBasis2DTransformer",
    "PCA",
    "main",
]


def main() -> None:
    print("Hello from geospectra!")
