# GeoSpectra

GeoSpectra provides a minimal set of tools for building polynomial and spherical
harmonics design matrices as well as a lightweight linear regression model.
All components follow the `scikit-learn` API so they can be composed in
pipelines and grid searches.

## Features

- **PolynomialBasis** — generate polynomial, Chebyshev or Legendre terms for
  two‑dimensional inputs.
- **SphericalHarmonicsBasis** — convert longitude and latitude to spherical
  harmonic features with multiple coordinate conversion options.
- **LinearRegressionCond** — linear regression enhanced with a configurable
  condition number threshold to improve numerical stability.
- **PCA** — simple principal component analysis for `pandas.DataFrame` inputs.
- Command line entry point `geospectra` that prints a greeting.

## Installation

Install the package and its development dependencies with
[`uv`](https://docs.astral.sh/uv/) or standard `pip`:

```bash
uv pip install .[dev]
# or
python -m pip install .
```

## Quickstart

```python
import numpy as np
from geospectra import PolynomialBasis, LinearRegressionCond

rng = np.random.default_rng(0)
X = rng.random((10, 2))
y = rng.random((10, 1))

basis = PolynomialBasis(degree=2)
design = basis.fit_transform(X)

reg = LinearRegressionCond()
reg.fit(design, y)

print(reg.predict(design))
```

## Development

Run the formatter, linter and tests:

```bash
ruff format .
ruff check .
pytest
```

GeoSpectra follows the SOLID principles while keeping the implementation
minimal, as described in `AGENTS.md`.
