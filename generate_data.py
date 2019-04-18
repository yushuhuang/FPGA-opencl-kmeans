import sklearn.datasets
import numpy as np

N = 1024
k = 3

data, labels = sklearn.datasets.make_blobs(
    n_samples=N, n_features=2, centers=k)
np.savetxt('data', data)
