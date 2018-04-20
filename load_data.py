import numpy as np


def createCompressedData(filename):
    # Create compressed binary dataset (Be comment out)
    from tensorflow.examples.tutorials.mnist import input_data
    print('Loading and extracting data...')
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_test = mnist.test.images
    y_test = mnist.test.labels
    np.savez_compressed(filename, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print('Created data.')

def loaddata(filename, N_train=None, N_val=None):
    N = 55000 # Original size of training dataset

    # Load data and pre-process label of image to label value from 0 to 9
    data = np.load(filename)

    # Default: N_train = 50000, N_val = 5000, N_test always = 10000
    if N_train is None:
        N_train = data['x_train'].shape[0] - 5000
    if N_val is None:
        N_val = 5000

    N_test = data['x_test'].shape[0]

    if N_train + N_val > N:
        print('Divide data for train and validation wrong. Check again !')
        quit()

    x_train = data['x_train'][:N_train, :].reshape(N_train,1,28,28)
    y_train = np.argwhere(data['y_train'][:N_train,:] == 1)
    y_train = y_train[:,1]

    x_val = data['x_train'][N_train:N_train+N_val].reshape(N_val, 1,28,28)
    y_val = np.argwhere(data['y_train'][N_train:N_train+N_val,:] == 1)
    y_val = y_val[:,1]

    x_test = data['x_test'].reshape(N_test, 1, 28, 28)
    y_test = np.argwhere(data['y_test'] == 1)
    y_test = y_test[:, 1]

    data = {
        'X_train': x_train, # training data
        'y_train': y_train, # training labels
        'X_val': x_val, # validation data
        'y_val': y_val, # validation labels
        'X_test': x_test, # test data
        'y_test': y_test # test labels
    }
    return data