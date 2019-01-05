# https://github.com/ctgk/PRML/blob/master/prml/linear/bayesian_regressor.py

import numpy as np
from ml.linear_model.linear_model import LinearModel

class BayesianLinearRegression(LinearModel):
    """
    Bayesian Linear Regression
    m : mean
    S^(-1) : precision
    m = S(S^(-1)*m + beta*X'*t)
    S^(-1) = alpha*I + beta*X'*X
    """

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.precision = None
        self.cov = None

    def _fit(self, X, y):
        if self.mean is not None:
            mean_prev = self.mean
        else:
            mean_prev = np.zeros(X.shape[1])
        if self.cov is not None:
            precision_prev = self.cov
        else:
            precision_prev = self.alpha * np.eye(X.shape[1])
        self.precision = precision_prev + self.beta * X.T @ X
        self.cov = np.linalg.inv(self.precision)
        self.mean = self.cov @ (precision_prev @ mean_prev + self.beta * X.T @ y)
        return self

    def _predict(self, X, return_std=False):
        y = np.dot(X, self.mean)
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y
