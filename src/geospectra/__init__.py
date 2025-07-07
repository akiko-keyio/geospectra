from .basis import CoordsConverter, PolynomialBasis, SphericalHarmonicsBasis
from .linear_model import BasisFunctionRegressor


def __getattr__(name: str):
    if name in {"LinearBasis2DTransformer", "PCA"}:
        from . import transform_coding

        obj = getattr(transform_coding, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
