import numpy as np


def generate_financial_data(num_samples=1000, num_features=50):
    np.random.seed(42)
    # add the covariance matrix definition and correlated feature generation here
    mean = np.zeros(num_features)
    cov_matrix = np.random.rand(num_features, num_features)
    cov_matrix = np.dot(cov_matrix, cov_matrix.T)  # Make it symmetric
    data = np.random.multivariate_normal(mean, cov_matrix, size=num_samples)
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)
    return data

if __name__ == "__main__":
    data = generate_financial_data()
    print(data.shape)