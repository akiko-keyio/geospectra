from .basis import CoordsConverter, PolynomialBasis, SphericalHarmonicsBasis
from .linear_model import BasisFunctionRegressor


def __getattr__(name: str):
    if name == "PCA":
        from . import transforms

        obj = getattr(transforms, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CoordsConverter",
    "PolynomialBasis",
    "SphericalHarmonicsBasis",
    "BasisFunctionRegressor",
    "PCA",
    "main",
]


def main() -> None:
    print("Hello from geospectra!")
