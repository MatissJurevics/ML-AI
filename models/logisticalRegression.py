import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticalRegression: 
    def __init__(self, learning_rate=0.005, n_itirs=1000):
        self.learning_rate = learning_rate
        self.n_itirs = n_itirs
        self.weigths = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weigths = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_itirs):
            y_pred = np.dot(X, self.weigths) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weigths = self.weigths - dw * self.learning_rate
            self.bias = self.bias - db * self.learning_rate
        
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weigths) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred