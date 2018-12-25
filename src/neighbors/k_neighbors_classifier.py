import numpy as np
from src.neighbors.kd_tree import KDTree

class KNeighborsClassifier():
    """
    K Nearest Neighbors Using KD Tree.
    """

    def _init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.tree = KDTree(X,y)

    def kneighbors(self, X):
        k_neighbors, _ = self.tree.query(X, self.n_neighbors)
        return neighbors

    def predict(self, X):
        _, labels = self.tree.query(X, self.n_neighbors)
        label, counts = np.unique(labels, return_counts=True)
        return label[np.argmax(counts)]
