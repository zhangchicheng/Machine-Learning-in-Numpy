import numpy as np
from src.linear_model.linear_model import LinearModel

class Ridge(LinearModel):
    """
    Ridge Regression Using Gradient Descent.

    Parameters
    ----------
    alpha : regularization strength
    eta : learning rate
    iter : num of times the parameters are updated
    coef_ : weights
    cost_ : average error of the model
    intercept_ : independent term
    """

    def __init__(self, alpha=0.01, eta=0.05, iter=1000):
        self.alpha = alpha
        self.eta = eta
        self.iter = iter

    def _fit(self, X, y):
        m, n = X.shape
        self.cost_ = np.zeros(self.iter)
        self.coef_ = np.zeros(n)
        self.intercept_ = 0.

        for i in range(self.iter):
            residues = np.dot(X, self.coef_) + self.intercept_ - y
            grad = np.dot(X.T, residues)
            self.coef_ -= self.eta / m * (grad + self.alpha * self.coef_)
            self.intercept_ -= self.eta / m * (np.sum(residues) + self.alpha * self.intercept_)
            self.cost_[i] = np.sum(residues ** 2) / (2 * m) + self.alpha / 2 * np.sum((self.coef_) ** 2)
        return self

    def _predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
