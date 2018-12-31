# https://xavierbourretsicotte.github.io/lasso_implementation.html

import numpy as np
from src.linear_model.linear_model import LinearModel

class Lasso(LinearModel):
    """
    Lasso Regression Using Coordinate Descent.

    Parameters
    ----------
    alpha : regularization strength
    eta : learning rate
    iter : num of times the parameters are updated

    Attributes
    ----------
    coef_ : weights
    cost_ : average error of the model
    intercept_ : independent term
    """

    def __init__(self, alpha=0.01, iter=1000):
        self.alpha = alpha
        self.iter = iter

    def _fit(self, X, y):
        m, n = X.shape
        self.cost_ = np.zeros(self.iter)
        self.coef_ = np.ones(n)
        self.intercept_ = 0.

        for i in range(self.iter):
            self.intercept_ = np.sum(y - np.dot(X, self.coef_)) / m
            for j in range(n):
                residues = y - (np.dot(X, self.coef_) + self.intercept_)
                rho = np.dot(X[:,j], (residues + self.coef_[j] * X[:,j]))
                if rho < -self.alpha:
                    self.coef_[j] = (rho + self.alpha) / np.sum(X[:,j] ** 2)
                elif rho > self.alpha:
                    self.coef_[j] = (rho - self.alpha) / np.sum(X[:,j] ** 2)
                else:
                    self.coef_[j] = 0
            residues = y - (np.dot(X, self.coef_) + self.intercept_)
            self.cost_[i] = np.sum(residues ** 2) / (2 * m) + self.alpha / 2 * np.sum(abs(self.coef_))
        return self

    def _predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
