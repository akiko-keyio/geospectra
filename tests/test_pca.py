import numpy as np
import pandas as pd
from geospectra.transforms import PCA


def test_pca_mean_correct():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(10, 3)), columns=list("abc"))
    pca = PCA()
    pca.fit(df)
    expected = df.to_numpy().mean(axis=0)
    assert np.allclose(pca.mean_, expected)
