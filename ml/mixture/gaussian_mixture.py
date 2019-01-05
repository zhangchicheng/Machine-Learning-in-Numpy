import numpy as np

class GaussianMixture():
    """
    Gaussian Mixture Model Parameter Estimation using EM algorithm

    Parameters
    ----------
    n_components : The number of mixture components.
    tol : The convergence threshold.
    max_iter : The number of EM iterations to perform.
    n_init : The number of initializations to perform. The best results are kept.

    Attributes
    ----------
    weights_ : The weights of each mixture components.
    means_ : The mean of each mixture component.
    covariances_ : The covariance of each mixture component.
    converged_ : True when convergence was reached in fit(), False otherwise.
    n_iter_ : Number of step used by the best fit of EM to reach the convergence.
    lower_bound_ : Lower bound value on the log-likelihood
    """

    def __init__(self, n_components=1, tol=0.001, max_iter=100, n_init=1):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init

    def fit(self, X):
        self.weights_ =  self.means_ = self.covariances_ = None
        self.converged_ = False
        self.n_iter = np.inf
        self.lower_bound_ = 0.
        self.m, self.n = X.shape

        trials = 0
        while trials < self.n_init:
            weights, means, covariances = self._init_params(X)
            converged = False
            iters = 0
            lower_bound = 0.
            while not converged and iters < self.max_iter:

                Q = self._expectation(X, weights, means, covariances)
                weights, means, covariances = self._maximization(X, Q)
                singular = False
                for i in range(self.n_components):
                    if np.any(np.isnan(covariances[i])) or np.linalg.cond(covariances[i]) > 1.e+8:
                        singular = True
                if singular:
                    break

                prob = 0.
                for j in range(self.n_components):
                    prob += self._mvnpdf(X, means[j], covariances[j]) * weights[j]
                prev_lower_bound = lower_bound
                lower_bound = np.sum(np.log(prob))

                if iters > 0:
                    converged = abs(lower_bound - prev_lower_bound) < self.tol

                iters += 1

            if converged and iters < self.n_iter:
                self.weights_ = weights
                self.means_ = means
                self.covariances_ = covariances
                self.converged_ = True
                self.n_iter = iters
                self.lower_bound_ = lower_bound

            trials += 1
        return self

    def predict(self, X):
        m, n = X.shape
        prob = np.zeros([m, self.n_components])
        for i in range(self.n_components):
            prob[:, i] = self._mvnpdf(X, self.means_[i], self.covariances_[i]) * self.weights_[i]
        return np.argmax(prob, axis=1)

    def _init_params(self, X):
        weights = np.ones(self.n_components) / self.n_components
        means = X[np.random.choice(self.m, self.n_components)]
        covariances = np.stack((np.identity(self.n) for _ in range(self.n_components)))
        return weights, means, covariances

    def _expectation(self, X, weights, means, covariances):
        Q = np.zeros([self.m, self.n_components])

        for i in range(self.n_components):
            Q[:, i] = self._mvnpdf(X, means[i], covariances[i]) * weights[i]

        Q = (Q.T / np.sum(Q, axis=1)).T
        return Q

    def _maximization(self, X, Q):
        weights = np.zeros(self.n_components)
        means = np.zeros([self.n_components, self.n])
        covariances = np.zeros([self.n_components, self.n, self.n])

        for i in range(self.n_components):
            weights[i] = np.mean(Q[:,i])
            means[i] = np.sum((X.T*Q[:,i]).T,axis=0) / np.sum(Q[:,i])
            for j in range(self.m):
                covariances[i] += Q[j][i] * np.reshape(X[j]-means[i], (-1,1)) * (X[j]-means[i]) / np.sum(Q[:,i])

        return weights, means, covariances

    def _mvnpdf(self, X, mean, cov):
        m, n = X.shape
        prob = np.zeros(m)
        coef = np.sqrt(np.power(2*np.pi, n)*abs(np.linalg.det(cov)))
        for i in range(m):
            prob[i] = np.exp(-0.5 * (X[i]-mean) @ np.linalg.inv(cov) @ (X[i]-mean)) / coef
        return prob
