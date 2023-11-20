from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from models.linearRegression import LinearRegression
from models.logisticalRegression import LogisticalRegression
from models.perceptron import Perceptron

# X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# line_reg = LinearRegression(learning_rate=0.01)
# line_reg.fit(X_train, y_train)
# y_pred = line_reg.predict(X_test)

# y_pred_line = line_reg.predict(X)

# plt.scatter(X, y)
# plt.plot(X, y_pred_line, color="blue")
# plt.show()

# bc = datasets.load_breast_cancer()

# X, y
#  = bc.data, bc.target
mnist = datasets.load_digits()
X,y = mnist.data, mnist.target
y = np.where(y==5, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

best_weight = None
best_bias = None
highest_acc = 0
# best_model = None

attempts = 20

def train_model(model, X_train, y_train, X_test, y_test):
    return 

for i in range(attempts):
    print(f'{i}/{attempts}:\tLoading Model\tBest Accuracy: {highest_acc}', end="\r")
    lr = 0.005
    if i % 10 == 0 & i != 0:
        print("\n\nAdjusting Learning rate...")
        lr * 2

    p_clf = Perceptron(learning_rate=0.005)
    p_clf.fit(X_train, y_train)
    y_pred = p_clf.predict(X_test)
    acc = accuracy(y_pred, y_test)
    if acc > highest_acc:
        best_weight, best_bias = p_clf.export_weights_and_biases()
        highest_acc = acc

p_clf = Perceptron(learning_rate=0.01)
p_clf.load_weights_and_biases(best_weight, best_bias)
y_pred = p_clf.predict(X_test)


acc = accuracy(y_pred, y_test)

def get_metrics(y_pred, y_test):
    acc = np.sum(y_pred==y_test)/len(y_test)
    recall = np.sum(y_pred[y_test==1]==1)/np.sum(y_test==1)
    precision = np.sum(y_pred[y_pred==1]==1)/np.sum(y_pred==1)
    return acc, recall, precision

acc, recall, precision = get_metrics(y_pred, y_test)
print("\n")
print(f'Accuracy: {acc}\nRecall: {recall}\nPrecision: {precision}')




