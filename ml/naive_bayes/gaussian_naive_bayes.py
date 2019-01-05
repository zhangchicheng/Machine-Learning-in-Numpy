import numpy as np

class GaussianNB():
    """
    Gaussian Naive Bayes
    """

    def __init__(self, priors=None):
        self.class_prior_ = priors

    def fit(self, X, y):
        """
        p(y|x1,x2,...,xn) = p(x1,x2,...,xn|y) * p(y)
        """
        m, n = X.shape
        if self.class_prior_ is None:
            self.label_, class_count_ = np.unique(y, return_counts=True)
            self.class_prior_ = class_count_ / m

        self.theta_ = np.zeros((len(self.label_), n))
        self.sigma_ = np.zeros((len(self.label_), n))
        for i in range(len(self.label_)):
            self.theta_[i, :] = np.mean(X[y==self.label_[i]], axis=0)
            self.sigma_[i, :] = np.var(X[y==self.label_[i]], axis=0)

        return self

    def predict(self, X):
        res = np.zeros(len(X))
        for i in range(len(X)):
            p = np.prod(self._likelihood(X[i], self.theta_, self.sigma_), axis=1) * self.class_prior_
            res[i] = self.label_[np.argmax(p)]
        return res

    def score(self, X, y):
        return np.sum(self.predict(X)==y) / len(y)

    def _likelihood(self, X, theta, sigma):
        return 1/ np.sqrt(2*np.pi*sigma) * np.exp(-(X-theta)**2 / (2*sigma))
