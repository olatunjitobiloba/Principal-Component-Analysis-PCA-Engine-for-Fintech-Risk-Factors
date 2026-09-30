import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


def generate_financial_data(num_samples=1000, num_features=50):
    np.random.seed(42)
    mean = np.zeros(num_features)
    cov_matrix = np.random.rand(num_features, num_features)
    cov_matrix = np.dot(cov_matrix, cov_matrix.T)  # Make it symmetric
    data = np.random.multivariate_normal(mean, cov_matrix, size=num_samples)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data


def load_data():
    uploaded = st.file_uploader("Upload your CSV", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        return data.select_dtypes("number")

    return generate_financial_data()


if __name__ == "__main__":
    data = generate_financial_data()
    print(data.shape)