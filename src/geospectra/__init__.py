from .basis import CoordsConverter, PolynomialBasis, SphericalHarmonicsBasis
from .linear_model import LinearRegressionCond


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
    "LinearRegressionCond",
    "PCA",
    "main",
]


def main() -> None:
    print("Hello from geospectra!")
