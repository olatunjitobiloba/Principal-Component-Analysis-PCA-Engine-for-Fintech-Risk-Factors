import numpy as np
import pytest

from pca_engine import PCAEngine

n_rows = 200
n_cols = 10

np.random.seed(0)
scales = np.arange(1, n_cols + 1)
X = np.random.normal(size=(n_rows, n_cols)) * scales


def test_components_shape():
    pca = PCAEngine(n_components=3)
    pca.fit_transform(X)

    components = pca.components
    assert components is not None
    assert components.shape == (n_cols, 3)


def test_scores_shape():
    pca = PCAEngine(n_components=3)
    scores = pca.fit_transform(X)

    assert scores.shape == (n_rows, 3)


def test_variance_is_descending():
    pca = PCAEngine(n_components=4)
    scores = pca.fit_transform(X)

    column_variances = np.var(scores, axis=0)

    for i in range(len(column_variances) - 1):
        assert column_variances[i] >= column_variances[i + 1]


def test_any_number_of_components():
    for k in [1, 2, 5, n_cols]:
        pca = PCAEngine(n_components=k)
        scores = pca.fit_transform(X)

        assert scores.shape == (n_rows, k)


def test_pca_orthogonality():
    pca = PCAEngine(n_components=n_cols)
    pca.fit_transform(X)
    components = pca.components
    assert components is not None
    pc1 = components[:, 0]
    pc2 = components[:, 1]
    assert abs(np.dot(pc1, pc2)) < 1e-7


def test_pca_normalized():
    pca = PCAEngine(n_components=n_cols)
    pca.fit_transform(X)
    components = pca.components
    assert components is not None
    pc1 = components[:, 0]
    assert np.linalg.norm(pc1) == pytest.approx(1.0, abs=1e-6)
