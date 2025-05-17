import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class TransformedRegressor(BaseEstimator, RegressorMixin):
    """A wrapper class that applies transformation to target variable before training and prediction.
    
    Args:
        regressor: The base regressor to use
        transform_fn: Function to transform target variable (default: np.log)
        inverse_fn: Function to inverse transform predictions (default: np.exp)
    """
    def __init__(self, regressor, transform_fn=np.log, inverse_fn=np.exp):
        self.regressor = regressor
        self.transform_fn = transform_fn
        self.inverse_fn = inverse_fn
    
    def fit(self, X, y):
        """Fit the model after transforming the target variable."""
        self.y_min_ = y.min()
        transformed_y = self.transform_fn(y)
        self.regressor.fit(X, transformed_y)
        return self
    
    def predict(self, X):
        """Make predictions and inverse transform them."""
        transformed_pred = self.regressor.predict(X)
        return self.inverse_fn(transformed_pred) 