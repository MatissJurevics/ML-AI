import numpy as np

def save_weights_and_biases(models):
    for i in range(10):
        weight, bias = models[i].export_weights_and_biases()
        
        np.save(f'utils/models/MNIST/{i}_weight.npy', weight)
        np.save(f'utils/models/MNIST/{i}_bias.npy', bias)


def load_weights_and_biases(models):
    for i in range(10):
        weight = np.load(f'utils/models/MNIST/{i}_weight.npy')
        bias = np.load(f'utils/models/MNIST/{i}_bias.npy')
        models[i].load_weights_and_biases(weight, bias)