import numpy as np
from perceptron import Perceptron
from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

class NeuralNet:
    def __init__(self,learning_rate=0.001, n_itirs=1000, multi_thread=False):
        self.learning_rate = learning_rate
        self.n_itirs = n_itirs
        self.data = None
        self.target = None
        self.target_unique = None
        self.nodeNum = 0
        self.nodes = {}
        self.multi_threaad = multi_thread
    
    def fit(self, data, target):
        self.data = data
        self.target = target
        self.target_unique = np.unique(target)
        self.nodeNum = self.target_unique.size

        if self.multi_threaad:
            threads = []
            print("Training Model...")
            with Pool(processes=self.nodeNum) as pool:
                for i in tqdm(range(self.nodeNum)):
                    node = Perceptron()
                    temp_target = np.where(self.target == self.target_unique[i], 1, 0)
                    threads.append(pool.apply_async(node.fit, (self.data, temp_target)))
                for i in tqdm(range(self.nodeNum)):
                    self.nodes[i] = threads[i].get()
        else:
            print("Training Model...")
            for i in tqdm(range(self.nodeNum)):
                node = Perceptron()
                temp_target = np.where(self.target == self.target_unique[i], 1, 0)
                node.fit(self.data, temp_target)
                self.nodes[i] = node
    
    def predict(self, data):
        confidence = np.zeros(self.nodeNum)

        for i in range(self.nodeNum):
            confidence[i] = self.nodes[i].predict_ovr(data)
        most_likely_value = np.argmax(confidence)
        return most_likely_value
        

    

    
if __name__ == "__main__":
    ds = fetch_olivetti_faces()
    # ds = load_digits()
    X,y = ds.data, ds.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    plt.imshow(X_train[0].reshape(64,64))
    plt.show()
    # nn = NeuralNet(multi_thread=True)
    # nn.fit(X_train, y_train)
    # acc = 0
    # print("Measuring Accuracy...")
    # for i in  tqdm(range(len(y_test))):
    #     y_pred = nn.predict(X_test[i])
    #     y_act = y_test[i]
    #     if y_pred == y_act:
    #         acc += 1


    # print("Accuracy: " + str((acc / len(y_test))))

    
