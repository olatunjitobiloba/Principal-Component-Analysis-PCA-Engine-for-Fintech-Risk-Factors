import numpy as np


class PCAEngine:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        indices = np.argsort(eigenvalues)[::-1][: self.n_components]

        self.components = eigenvectors[:, indices]

        self.explained_variance = eigenvalues[indices]

        return self.explained_variance

    def fit_transform(self, X):
        self.fit(X)

        return self.transform(X)

    def transform(self, X):
        X_centered = X - self.mean

        return X_centered @ self.components

    def inverse_transform(self, scores):
        components = self.components
        assert components is not None
        X_reduced = scores @ components.T

        return X_reduced + self.mean

    def reconstruction_error(self, X):
        X_reconstructed = self.inverse_transform(self.transform(X))

        return np.mean((X - X_reconstructed) ** 2)
