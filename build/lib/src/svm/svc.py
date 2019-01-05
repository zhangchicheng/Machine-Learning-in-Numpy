import numpy as np

class SVC():
    """
    Support Vector Classification using SMO
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, max_iter=-1):
        self.C = C
        self.kernel = 'rbf'
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        if max_iter < 0:
            self.max_iter = np.inf
        else:
            self.max_iter = max_iter

    def _init_args(self, X, y):
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.alpha = np.ones(self.m)
        self.b = 0.0

        if self.gamma == 'auto':
            self.gamma = 1 / self.n
        elif self.gamma == 'scale':
            self.gamma = 1/ (self.n * X.std())
        else:
            raise NameError

        self.E = [self._E(i) for i in range(self.m)]

    def _K(self, x1, x2):
        if self.kernel == 'linear':
            return x1@x2
        elif self.kernel == 'poly':
            return (self.gamma * x1@x2 + self.coef0)**self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * (x1-x2) @ (x1-x2))
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * x1@x2 + self.coef0)
        else:
            raise NameError

    def _E(self, i):
        res = -self.b
        for j in range(self.m):
            res += self.y[j] * self.alpha[j] * self._K(self.X[i], self.X[j])
        return res - self.y[i]

    def _chooseMultiplier(self, i1):
        # section 2.2
        non_bound = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        bound= [i for i in range(self.m) if i not in non_bound]
        non_bound.extend(bound)

        max_step = 0.0
        for i in non_bound:
            if i1 == i:
                continue
            if abs(self.E[i1] - self.E[i]) > max_step:
                max_step = abs(self.E[i1] - self.E[i])
                i2 = i
        return i2

    def _computeLH(self, i1, i2):
        # equation (13) and (14)
        if self.y[i1] == self.y[i2]:
            L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
            H = min(self.C, self.alpha[i2] + self.alpha[i1])
        else:
            L = max(0, self.alpha[i2] - self.alpha[i1])
            H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
        return L, H


    def _examine_example(self, i1):
        if (self.y[i1]* self.E[i1] < -self.tol) and (self.alpha[i1] < self.C) or (self.y[i1] * self.E[i1] > self.tol) and (self.alpha[i1] > 0):
            i2 = self._chooseMultiplier(i1)
            L, H = self._computeLH(i1, i2)
            if L == H:
                return 0

            # equation (15)
            eta = self._K(self.X[i1], self.X[i1]) + self._K(self.X[i2], self.X[i2]) - 2*self._K(self.X[i1], self.X[i2])
            if eta <= 0:
                return 0
            # equation (16)
            alpha2_new = self.alpha[i2] + self.y[i2] * (self.E[i1] - self.E[i2]) / eta

            # equation (17)
            if alpha2_new >= H:
                alph2_new = H
            elif alpha2_new <= L:
                alph2_new = L

            if abs(alpha2_new-self.alpha[i2]) < self.tol:
                return 0

            # equation (18)
            s = self.y[i1] * self.y[i2]
            alph1_new = self.alpha[i1] + s * (self.alpha[i2] - alph2_new)

            # equation (20)
            b1 = self.E[i1] + self.y[i1] * (alph1_new - self.alpha[i1]) * self._K(self.X[i1], self.X[i1]) + \
                 self.y[i2] * (alph2_new - self.alpha[i2]) * self._K(self.X[i1], self.X[i2]) + self.b
            # equation (21)
            b2 = self.E[i2] + self.y[i1] * (alph1_new - self.alpha[i1]) * self._K(self.X[i1], self.X[i2]) + \
                 self.y[i2] * (alph2_new - self.alpha[i2]) * self._K(self.X[i2], self.X[i2]) + self.b

            if self.tol < alph1_new < self.C-self.tol:
                b_new = b1
            elif self.tol < alph2_new < self.C-self.tol:
                b_new = b2
            else:
                b_new = (b1 + b2) / 2.0

            self.alpha[i1] = alph1_new
            self.alpha[i2] = alph2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)

            return 1
        else:
            return 0

    def fit(self, X, y):
        self._init_args(X, y)
        num_changed = 0
        examine_all = True
        iters = 0

        while iters < self.max_iter and num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.m):
                    num_changed += self._examine_example(i)
            else:
                non_zero_C = np.nonzero((self.alpha>self.tol) * (self.alpha<self.C-self.tol))[0]
                for i in non_zero_C:
                    num_changed += self._examine_example(i)
            iters += 1

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        return self

    def predict(self, X):
        res = -self.b
        for i in range(self.m):
            res += self.alpha[i] * self.y[i] * self._K(X, self.X[i])
        if res > 0:
            return 1
        else:
            return -1

    def score(self, X, y):
        cnt = 0
        for i in range(len(X)):
            res = self.predict(X[i])
            if res == y[i]:
                cnt += 1
        return cnt / len(X)
