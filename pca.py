import numpy as np


class PCAEngine:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit_transform(self, X):
        # center the data by subtracting the mean along axis 0
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # compute covariance matrix, eigenvalues, and eigenvectors
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvalues and eigenvectors in descending order
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # get the top k eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # project centered data onto the top k eigenvectors
        return X_centered @ self.components