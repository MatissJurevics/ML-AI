import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_itirs=1000):
        self.learning_rate = learning_rate
        self.n_itirs = n_itirs
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_itirs):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - dw * self.learning_rate
            self.bias = self.bias  - db * self.learning_rate

    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
