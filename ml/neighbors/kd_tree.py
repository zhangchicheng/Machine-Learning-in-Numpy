import numpy as np

class KDTree():
    """
    KDTree
    """

    def __init__(self, X, y=None):
        pivot = np.argmax(np.var(X, axis=0))
        idx = np.argsort(X[:,pivot])
        median = len(idx) // 2

        self.val = X[idx[median]]
        self.pivot = pivot
        self.label = None
        self.left = None
        self.right = None

        if y is not None:
            self.label = y[idx[median]]
        if X[idx[:median]].size != 0:
            self.left = KDTree(X[idx[:median]], y=y[idx[:median]] if y is not None else None)
        if X[idx[median+1:]].size != 0:
            self.right = KDTree(X[idx[median+1:]], y=y[idx[median+1:]] if y is not None else None)

    def query(self, X, n_neighbors=1):
        max_dist = 0.
        Nodes = []
        cur = self
        NearestNeighbors = None
        Nodes.append(cur)

        # inorder traversal
        while Nodes:
            while cur is not None:
                if X[cur.pivot] < cur.val[cur.pivot]:
                    if cur.left is not None:
                        Nodes.append(cur.left)
                    cur = cur.left
                else:
                    if cur.right is not None:
                        Nodes.append(cur.right)
                    cur = cur.right

            cur = Nodes.pop()
            dist = np.linalg.norm(cur.val-X)

            # push node to NN if necessary
            if NearestNeighbors is None:
                NearestNeighbors = np.array([cur.val])
                Labels = np.array([cur.label])
                max_dist = dist
            else:
                if len(NearestNeighbors)<n_neighbors:
                    NearestNeighbors = np.vstack((NearestNeighbors, cur.val))
                    Labels = np.append(Labels, cur.label)
                    idx = np.argsort(np.linalg.norm(NearestNeighbors-X, axis=1))
                    NearestNeighbors = NearestNeighbors[idx]
                    Labels = Labels[idx]
                    max_dist = max(max_dist, dist)
                elif dist < max_dist:
                    NearestNeighbors[-1] = cur.val
                    max_dist = dist

            # determine if search another branch
            if abs(X[cur.pivot] - cur.val[cur.pivot]) <= max_dist or len(NearestNeighbors)<n_neighbors:
                if X[cur.pivot] < cur.val[cur.pivot]:
                    if cur.right is not None:
                        Nodes.append(cur.right)
                    cur = cur.right
                else:
                    if cur.left is not None:
                        Nodes.append(cur.left)
                    cur = cur.left

        return NearestNeighbors, Labels
