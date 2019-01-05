import numpy as np

class LogisticRegression():
    """
    Logistic Regression
    """

    def __init__(self, alpha=0.1, iter=300000):
        self.alpha = alpha
        self.iter = iter

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        self.intercept_ = np.ones((X.shape[0], 1))
        X = np.concatenate((self.intercept_, X), axis=1)
        self.coef_ = np.zeros(X.shape[1])

        for i in range(self.iter):
            z = np.dot(X, self.coef_)
            h = self._sigmoid(z)
            grad = np.dot(X.T, (h - y)) / y.size
            self.coef_ -= self.alpha * grad

    def predict_proba(self, X):
        X = np.insert(X, 0, 1)
        return self._sigmoid(np.dot(X, self.coef_))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold
