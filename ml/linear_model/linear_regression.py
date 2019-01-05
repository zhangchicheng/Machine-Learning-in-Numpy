import numpy as np
from ml.linear_model.linear_model import LinearModel

class LinearRegression(LinearModel):
    """
    Linear Regression Using Gradient Descent.

    Parameters
    ----------
    eta : learning rate
    iter : num of times the parameters are updated

    Attributes
    ----------
    coef_ : weights
    cost_ : average error of the model
    intercept_ : independent term
    """

    def __init__(self, eta=0.01, iter=1000):
        self.eta = eta
        self.iter = iter

    def _fit(self, X, y):
        m, n = X.shape
        self.cost_ = np.zeros(self.iter)
        self.coef_ = np.ones(n)
        self.intercept_ = 0.

        for i in range(self.iter):
            residues = np.dot(X, self.coef_) + self.intercept_ - y
            grad = np.dot(X.T, residues)
            self.coef_ -= self.eta * grad / m
            self.intercept_ -= self.eta * np.sum(residues) / m
            self.cost_[i] = np.sum((residues ** 2)) / (2 * m)
        return self

    def _predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
