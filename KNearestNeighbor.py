from statistics import mode
import numpy as np
from sklearn.metrics import DistanceMetric

"""
K-Nerest Neighbors implementation

You can choose the desire distance metric, for more information, take a look at
the 'get_metric' method of class 'DistanceMetric' in library scikit-learn
"""

class KNN():
    def __init__(self, X, Y, d='minkowski'):
        self.X = X
        self.Y = Y
        self.car, self.dim = X.shape
        self.dist = DistanceMetric.get_metric(d)
    
    def most_frequent(self, lst):
        return mode(lst)

    def predict(self, X_test, nn=3):
        label = []
        for i in range(X_test.shape[0]):
            dis = []
            for j in range(self.car):
                dd = self.dist.pairwise([X_test[i]], [self.X[j]])[0][0]
                dis.append((dd, self.Y[j]))
            
            # Sort distances and get the nearest nn
            dis.sort(key=lambda x: x[0])
            nearest_neighbors = [float(y) for _, y in dis[:nn]]  # List of labels
            
            # Get the most frequent label among the nearest neighbors
            label.append(self.most_frequent(nearest_neighbors))
        
        return np.array(label)