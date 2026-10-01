import numpy as np
import streamlit as st

from data_gen import load_data
from pca_engine import PCAEngine

st.title("Fintech Risk Factor PCA Engine")

data = load_data()

num_features = data.shape[1]

k = st.slider("Number of components (k)", 1, num_features, 5)

pca = PCAEngine(n_components=k)
explained_variance = pca.fit(data)

ratio = explained_variance / np.sum(explained_variance)

st.bar_chart(ratio)
st.caption(f"Explained variance ratio for each of the {k} selected components")

cumulative = np.cumsum(ratio)
selected = cumulative[-1]

error = pca.reconstruction_error(data)

st.write(f"{k} components explain {selected:.2%} of total variance")
st.write(f"Data reduced from {num_features} features to {k} components")
st.write(f"Reconstruction error (MSE): {error:.6f}")
