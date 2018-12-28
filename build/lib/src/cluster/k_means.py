import numpy as np

class KMeans():
    """
    KMeans
    """

    def __init__(self, n_clusters=2, n_init=10, tol=0.0001):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.tol = tol

    def fit(self, X):
        self.cluster_centers = np.empty((self.n_clusters, X.shape[1]))
        self.labels_ = np.empty(self.n_clusters)
        self.inertia_ = np.inf

        for i in range(self.n_init):
            centers = X[np.random.choice(len(X), self.n_clusters)]
            diff = np.inf
            while diff > self.tol:
                dist = np.dot(np.square(centers), np.ones(X.T.shape)) + \
                       np.dot(np.ones(centers.shape), np.square(X.T)) - \
                       2*np.dot(centers, X.T)
                labels = np.argmin(dist, axis=0)

                if len(np.unique(labels)) != self.n_clusters:
                    break

                pre_centers = centers
                for j in range(self.n_clusters):
                    centers[j] = np.mean(X[labels==j], axis=0)

                diff = max(np.linalg.norm(centers-pre_centers, axis=1))

            if np.sum(np.min(dist, axis=0)) < self.inertia_:
                self.labels_ = labels
                self.cluster_centers_ = centers
                self.inertia_ = np.sum(np.min(dist, axis=0))
        return self

    def predict(self, X):
        return np.argmin(np.linalg.norm(X-self.cluster_centers_, axis=-1))
