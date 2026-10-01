import matplotlib

matplotlib.use("Agg")

import numpy as np
import streamlit as st

from data_gen import load_data
from pca_engine import PCAEngine
from visualization import plot_cumulative_explained_variance, plot_explained_variance

st.title("Fintech Risk Factor PCA Engine")

data = load_data()

num_features = data.shape[1]

k = st.slider("Number of components (k)", 1, num_features, 5)

pca = PCAEngine(n_components=num_features)
explained_variance = pca.fit(data)
assert explained_variance is not None

ratio = explained_variance / np.sum(explained_variance)
cumulative = np.cumsum(ratio)

st.subheader("Explained Variance Ratio")
st.pyplot(plot_explained_variance(explained_variance))

st.subheader("Cumulative Explained Variance")
st.pyplot(plot_cumulative_explained_variance(explained_variance))

selected = cumulative[k - 1]

st.write(f"{k} components explain {selected:.2%} of total variance")
st.write(f"Data reduced from {num_features} features to {k} components")
st.write(f"Reconstruction error (MSE): {pca.reconstruction_error(data):.6f}")
