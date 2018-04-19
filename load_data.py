import numpy as np


# Create compressed binary dataset (Be comment out)

# from tensorflow.examples.tutorials.mnist import input_data\
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# x_train = mnist.train.images
# y_train = mnist.train.labels
# x_val = mnist.test.images
# y_val = mnist.test.labels
# np.savez_compressed('dataset.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

# Load data and pre-process label of image to label value from 0 to 9

def loaddata(filename, N_train=None, N_val=None):
    data = np.load(filename)

    if N_train is None:
        N_train = data['x_train'].shape[0]
    if N_val is None:
        N_val = data['x_val'].shape[0]

    x_train = data['x_train'][:N_train, :].reshape(N_train,1,28,28)
    y_train = np.argwhere(data['y_train'][:N_train,:] == 1)
    y_train = y_train[:,1]

    x_val = data['x_val'][:N_val].reshape(N_val, 1,28,28)
    y_val = np.argwhere(data['y_val'][:N_val,:] == 1)
    y_val = y_val[:,1]

    data = {
        'X_train': x_train, # training data
        'y_train': y_train, # training labels
        'X_val': x_val, # validation data
        'y_val': y_val # validation labels
    }
    return data