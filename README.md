# PCA Engine for Fintech Risk Factors

A Principal Component Analysis engine built from scratch with NumPy, for reducing
correlated financial risk factor data to a small number of independent components.
No `sklearn.decomposition.PCA` is used, so every step of the eigendecomposition is
visible and testable.

## Why it exists

Fintech risk datasets are wide and redundant. Fifty market indicators often collapse
to a handful of underlying factors, because most indicators move together. PCA finds
those underlying axes so a model can use 5 components instead of 50 columns.

## Requirements

Python 3.13 and the packages in `requirements.txt`: NumPy, scikit-learn (only for
`StandardScaler`), pandas, pytest, streamlit, matplotlib.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the dashboard

```powershell
streamlit run dashboard.py
```

Open the local URL it prints. Upload a CSV of numeric risk indicators, or leave the
uploader empty to fall back to 50 synthetic correlated indicators. Drag the slider to
choose how many components to keep.

## Running the tests

```powershell
python -m pytest test_pca.py -v
```

Ten tests cover output shapes, descending explained variance, arbitrary component
counts, orthogonality, unit norm, and reconstruction behaviour.

## Usage as a library

```python
from data_gen import generate_financial_data
from pca_engine import PCAEngine

data = generate_financial_data(num_samples=1000, num_features=50)

pca = PCAEngine(n_components=5)
pca.fit(data)

scores = pca.transform(data)          # reduced data, shape (1000, 5)
rebuilt = pca.inverse_transform(scores)  # back to 50 columns
error = pca.reconstruction_error(data)   # MSE, 0.0 when using all components
```

## Architecture

| File | Role |
|---|---|
| `pca_engine.py` | `PCAEngine`: mean centering, covariance, `np.linalg.eigh`, projection, reconstruction |
| `pca.py` | compatibility shim re-exporting `PCAEngine`, so `from pca import PCAEngine` also works |
| `data_gen.py` | synthetic correlated data, and `load_data()` for the dashboard |
| `visualization.py` | matplotlib charts for explained variance ratio and its cumulative form |
| `dashboard.py` | Streamlit UI wiring the pieces together |
| `test_pca.py` | pytest suite |

Data generation, the engine, and plotting are kept apart so each can be tested on its
own. The engine has no knowledge of Streamlit.

## How it works

1. Subtract the column mean from every row.
2. Compute the covariance matrix with `np.cov`.
3. Call `np.linalg.eigh`, which returns eigenvalues and eigenvectors.
4. Sort eigenvalues descending and keep the top `n_components` eigenvectors.
5. Project the centered data onto them with a matrix multiplication.

Because `np.linalg.eigh` returns orthonormal eigenvectors, the components are
orthogonal and unit length by construction. The tests verify this rather than assume it.

## Explaining the charts

`Explained Variance Ratio` is each component's share of total variance, so the bars
sum to 1.0. `Cumulative Explained Variance` is the running total, so the curve ends at
1.0. To reach 95% of variance, find where that curve crosses 0.95. That count is how
many components you keep.

## Standardization

The synthetic data is standardized with `StandardScaler` before analysis. PCA finds
the direction of greatest spread, and spread is measured in each feature's own units,
so an indicator stored in large units would otherwise dominate every component.
Standardizing puts all features on a scale of one.

## Docker

```powershell
docker build -t pca-engine .
docker run -p 8501:8501 pca-engine
```

## CI

`.github/workflows/tests.yml` runs the test suite on every push and pull request to
`main`.

## Known limitations

- Reconstruction error is MSE across all cells, not a variance-weighted measure.
- `load_data()` keeps only numeric columns and drops all others silently.
- The covariance matrix is computed densely, so this suits tens to low hundreds of
  features, not tens of thousands.
