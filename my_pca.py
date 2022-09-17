import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

data = pd.read_csv('E:/研一上/gitlearn/code_ppt/WZU-machine-learning-course/code/14-降维/data/pcadata.csv')
data.head(10)
X = data.values
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V


U, S, V = pca(X)


def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)


Z = project_data(X, U, 1)


def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)


X_recovered = recover_data(Z, U, 1)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()
faces = loadmat('E:/研一上/gitlearn/code_ppt/WZU-machine-learning-course/code/14-降维/data/ex7faces.mat')
X = faces['X']
X.shape
face = np.reshape(X[3,:], (32, 32))
plt.imshow(face)
plt.show()
U, S, V = pca(X)
Z = project_data(X, U, 100)
X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[3,:], (32, 32))
plt.imshow(face)
plt.show()
