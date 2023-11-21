import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from models.perceptron import Perceptron
from tqdm import tqdm

models = {}

mnist = load_digits()
X,y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


for i in tqdm(range(10)):
    models[i] = Perceptron(learning_rate=0.005)
    models[i].fit(X_train, np.where(y_train==i, 1, 0))

predictions = []

for i in tqdm(range(10)):
    predictions.append([models[i].predict_ovr(X_test)])

y_pred = np.zeros(len(X_test))

for i in tqdm(range(len(X_test))):
    y_pred[i] = np.argmax([predictions[j][0][i] for j in range(10)])
    

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

print(accuracy(y_pred, y_test))

