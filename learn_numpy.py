from numpy import matrix
import numpy as np
from scipy import misc
from layers import *

from lenet5_model import *
from train import *
from tensorflow.examples.tutorials.mnist import input_data\


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# x_train = mnist.train.images
# y_train = mnist.train.labels
# x_val = mnist.test.images
# y_val = mnist.test.labels
#
# Pre-processing data
# x_train = x_train.reshape(x_train.shape[0],1,28,28)
# y_train = np.argwhere(y_train[:,:] == 1)
# y_train = y_train[:,1]
#
# x_val = x_val.reshape(x_val.shape[0], 1,28,28)
# y_val = np.argwhere(y_val[:,:] == 1)
# y_val = y_val[:,1]
#
# data = {
#     'X_train': x_train, # training data
#     'y_train': y_train, # training labels
#     'X_val': x_val, # validation data
#     'y_val': y_val # validation labels
# }
#
# model = Lenet5ConvNet(input_dim=(1,28,28),
#                       weight_scale=1e-3, reg=10.0,
#                       dtype=np.float32)
#
# solver = Solver(model, data,
#                 update_rule='sgd',
#                 optim_config={
#                   'learning_rate': 0.1,
#                 },
#                 lr_decay=1,  #After each update, regurlarization decrease
#                 batch_size=10,
#                 num_epochs=20,
#                 num_train_samples=10000,
#                 num_val_samples=2000,
#                 print_every=100,
#                 checkpoint_name=None,
#                 verbose=True)
# solver.train()
