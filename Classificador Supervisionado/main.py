import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import pandas as pd

class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        dist = euclidean_distances(X, self.X_)
        closest = np.argsort(dist, axis=1)[:, :self.n_neighbors]
        y_closest = self.y_[closest]

        # Para cada conjunto de vizinhos mais próximos, encontra a classe mais comum
        y_pred = np.array([np.argmax(np.bincount(y)) for y in y_closest])

        return y_pred

base = pd.read_csv('Dmoz-Sports.csv')
base.head()