import numpy as np

class DecisionTreeClassifier():
    """
    Decision Tree Classifier using ID3
    """

    def __init__(self):
        pass

    def fit(self, X, t):
        # mask: determine if the feature has been processed
        mask = np.ones(X.shape[1], dtype=bool)
        self.tree = self._ID3(X, t, mask)
        return self

    def predict(self, X):
        return self._predict(self.tree, X)

    def _ID3(self, X, t, mask):
        if len(np.unique(t))==1:
            return t[0]
        n_feat = X.shape[1]
        info_gain = np.zeros(n_feat)

        for i in range(n_feat):
            if mask[i]:
                info_gain[i] = self._info_gain(X[:,i],t)

        idx = np.argmax(info_gain)
        features = np.unique(X[:,idx])
        tree = {}
        tree[idx] = {}
        mask[idx] = False
        for val in features:
            subX, subt = self._split(idx, val, X, t)
            tree[idx][val] = self._ID3(subX, subt, mask)
        return tree

    def _entropy(self, target):
        labels, counts = np.unique(target, return_counts=True)
        prob = [cnt/np.sum(counts) for cnt in counts]
        entropy =  np.sum([-p*np.log2(p) for p in prob])
        return entropy

    def _info_gain(self, feature, target):
        entropy = self._entropy(target)
        value, counts = np.unique(feature, return_counts=True)
        prob = [cnt/np.sum(counts) for cnt in counts]

        for i in range(len(value)):
            entropy -= prob[i]*self._entropy(target[feature==value[i]])
        return entropy

    def _split(self, idx, val, X, t):
        i = X[:,idx]==val
        return X[i], t[i]

    def _predict(self, tree, X):
        idx = list(tree.keys())[0]
        branch = X[idx]
        subTree = tree[idx][branch]
        if isinstance(subTree, dict):
            return self._predict(subTree, X)
        else:
            return subTree
