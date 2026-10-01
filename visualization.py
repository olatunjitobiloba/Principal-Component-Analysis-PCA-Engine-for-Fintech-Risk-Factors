import matplotlib.pyplot as plt
import numpy as np


def plot_explained_variance(eigenvalues):
    """Bar chart of the explained variance ratio for each component."""
    eigenvalues = np.asarray(eigenvalues)

    ratio = eigenvalues / np.sum(eigenvalues)

    figure, axis = plt.subplots()

    axis.bar(range(1, len(ratio) + 1), ratio, color="#1f77b4")

    axis.set_xlabel("Principal component")
    axis.set_ylabel("Explained variance ratio")
    axis.set_title("Explained Variance Ratio by Component")
    axis.set_xticks(range(1, len(ratio) + 1))

    return figure


def plot_cumulative_explained_variance(eigenvalues):
    """Line chart of the cumulative explained variance ratio."""
    eigenvalues = np.asarray(eigenvalues)

    cumulative = np.cumsum(eigenvalues / np.sum(eigenvalues))

    figure, axis = plt.subplots()

    axis.plot(range(1, len(cumulative) + 1), cumulative, marker="o", color="#1f77b4")

    axis.set_xlabel("Number of components")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_title("Cumulative Explained Variance")
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha=0.3)

    return figure
