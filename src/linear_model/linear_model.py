"""
Generalized Linear models.
"""

class LinearModel():
    """
    Base class for linear model
    """

    def fit(self, X, y, **kwargs):
        """
        X: [size, features] input
        y: [size, ] target
        """
        if hasattr(self, "_fit"):
            return self._fit(X, y, **kwargs)
        else:
            raise NotImplementedError

    def predict(self, X, **kwargs):
        """
        predict outputs of the linear_model
        """
        if hasattr(self, "_predict"):
            return self._predict(X, **kwargs)
        else:
            raise NotImplementedError
