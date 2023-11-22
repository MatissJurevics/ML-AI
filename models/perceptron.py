import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.001, n_itirs=1000):
        self.learning_rate = learning_rate
        self.n_itirs = n_itirs
        self.weights = None
        self.bias = None

    def activation_function(self, x):
        return np.where(x>0, 1,0)

    def fit(self, X, y):
        # Initialising weights
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=n_features)
        self.bias = np.random.uniform(low=-0.5, high=0.5)

        # Training the model
        for _ in range(self.n_itirs):
            for example_index, example_features in enumerate(X):
                y_input = np.dot(example_features, self.weights) + self.bias
                y_pred = self.activation_function(y_input)
                error = y[example_index] - y_pred
                adjusted_error = error * self.learning_rate
                self.weights = self.weights + adjusted_error * example_features
                self.bias = adjusted_error

    def predict(self, X):
        y_input = np.dot(X, self.weights) + self.bias
        y_pred = np.where(y_input > 0, 1, 0)
        return y_pred
    
    def predict_ovr(self, X): # One vs Rest prediction
        y_input = np.dot(X, self.weights) + self.bias
        return y_input
    
    def export_weights_and_biases(self):
        return (self.weights, self.bias)
    
    def load_weights_and_biases(self, weights, bias):
        self.weights = weights
        self.bias = bias
