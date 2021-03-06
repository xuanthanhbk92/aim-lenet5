import numpy as np

from layer_utils import *
from layers import *

class Lenet5ConvNet(object):
    def __init__(self, input_dim=(1,28,28), num_conv1=6, conv1_size=5,
                 num_conv2=16, conv2_size=5, dim_fc1=120, dim_fc2=84, num_classes=10,
                 dropout=0, seed=None,
                 weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.use_dropout = dropout > 0
        self.dtype = dtype

        c, h, w = input_dim
        self.input_dim = input_dim

        # Weights of CONV1: 6x5x5
        self.params['W1'] = weight_scale*np.random.randn(num_conv1, c, conv1_size, conv1_size)
        self.params['b1'] = np.zeros(num_conv1)

        # Weights of CONV2: 16x6x5x5
        self.params['W2'] = weight_scale*np.random.randn(num_conv2, num_conv1, conv2_size, conv2_size)
        self.params['b2'] = np.zeros(num_conv2)

        h_in_fc1 = 5
        w_in_fc1 = 5
        # Weight of FC1: 400x120
        self.params['W3'] = weight_scale * np.random.randn(num_conv2 * h_in_fc1 * w_in_fc1, dim_fc1)
        self.params['b3'] = np.zeros(dim_fc1)

        # Weight of FC2: 120x84
        self.params['W4'] = weight_scale * np.random.randn(dim_fc1, dim_fc2)
        self.params['b4'] = np.zeros(dim_fc2)

        # Weight of FC3: 84x10
        self.params['W5'] = weight_scale * np.random.randn(dim_fc2, num_classes)
        self.params['b5'] = np.zeros(num_classes)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None, mode='train'):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # Forward propagation

        # pass conv_param to the forward pass for the convolutional layer
        conv1_param = {'stride': 1, 'pad': 2}
        conv2_param = {'stride': 1, 'pad': 0}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode

        # CONV1-RELU-POOLING: Input=Nx1x28x28, Output = Nx6x14x14
        A1, conv1_cache = conv_relu_pool_forward(X, W1, b1, conv1_param, pool_param)
        if self.use_dropout and mode == 'train':
            A1, dropA1_cache = dropout_forward(A1, self.dropout_param)

        # CONV2-RELU-POOLING: Input=Nx6x14x14, Output = Nx16x5x5
        # A2, conv2_cache = conv_relu_pool_forward(A1, W2, b2, conv2_param, pool_param)
        A2, conv2_cache = conv_relu_pool_forward(A1, W2, b2, conv2_param, pool_param)
        if self.use_dropout and mode == 'train':
            A2, dropA2_cache = dropout_forward(A2, self.dropout_param)

        # FC1-RELU: Input=Nx16x5x5, Output = Nx120
        # A3, fc1_cache = affine_relu_forward(A2, W3, b3)
        A3, fc1_cache = affine_relu_forward(A2, W3, b3)
        if self.use_dropout and mode == 'train':
            A3, dropA3_cache = dropout_forward(A3, self.dropout_param)

        # FC2-RELU: Input=Nx120, Output = Nx84
        # A4, fc2_cache = affine_relu_forward(A3, W4, b4)
        A4, fc2_cache = affine_relu_forward(A3, W4, b4)
        if self.use_dropout and mode == 'train':
            A4, dropA4_cache = dropout_forward(A4, self.dropout_param)

        # FC3: Input=Nx84, Output = Nx10
        # scores, fc3_cache = affine_forward(A4, W5, b5)
        scores, fc3_cache = affine_forward(A4, W5, b5)

        if y is None:
            return scores

        loss, grads = 0, {}

        # Input = Nx1, Output = Nx10
        loss, dloss = cross_entropy(scores, y)

        # Backward propagation
        # FC3 backward: Input=Nx10, Output = Nx84
        dfc3, grads['W5'], grads['b5'] = affine_backward(dloss, fc3_cache)

        # FC2-RELU backward: Input=Nx84, Output = Nx120
        # dfc2, grads['W4'], grads['b4'] = affine_relu_backward(dfc3, fc2_cache)
        if self.use_dropout and mode == 'train':
            dfc3 = dropout_backward(dfc3, dropA4_cache)
        dfc2, grads['W4'], grads['b4'] = affine_relu_backward(dfc3, fc2_cache)

        # FC1-RELU backward: Input=Nx120, Output = Nx16x5x5
        # dfc1, grads['W3'], grads['b3'] = affine_relu_backward(dfc2, fc1_cache)
        if self.use_dropout and mode == 'train':
            dfc2 = dropout_backward(dfc2, dropA3_cache)
        dfc1, grads['W3'], grads['b3'] = affine_relu_backward(dfc2, fc1_cache)

        # CONV2-RELU-POOLING backward: Input=Nx16x5x5, Output = Nx6x14x14
        # dconv2, grads['W2'], grads['b2'] = conv_relu_pool_backward(dfc1, conv2_cache)
        if self.use_dropout and mode == 'train':
            dfc1 = dropout_backward(dfc1, dropA2_cache)
        dconv2, grads['W2'], grads['b2'] = conv_relu_pool_backward(dfc1, conv2_cache)

        # CONV1-RELU-POOLING backward: Input=Nx6x14x14, Output = Nx1x28x28
        # dconv1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dconv2, conv1_cache)
        if self.use_dropout and mode == 'train':
            dconv2 = dropout_backward(dconv2, dropA1_cache)
        dconv1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dconv2, conv1_cache)

        # Add L2 regularization
        loss += 0.5 * self.reg * (
        np.square(W1).sum() + np.square(W2).sum() + np.square(W3).sum() + np.square(W4).sum() + np.square(W5).sum())

        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        grads['W3'] += self.reg * self.params['W3']
        grads['W4'] += self.reg * self.params['W4']
        grads['W5'] += self.reg * self.params['W5']

        return loss, grads


class Lenet5Simple(object):
    def __init__(self, input_dim=(1,28,28), num_conv1=6, conv1_size=3,
                 num_conv2=16, conv2_size=5, dim_fc1=120, dim_fc2=84, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        c, h, w = input_dim

        self.input_dim = input_dim

        # Weights of CONV1: 6x5x5
        self.params['W1'] = weight_scale*np.random.randn(num_conv1, c, conv1_size, conv1_size)
        self.params['b1'] = np.zeros(num_conv1)

        h_in_fc1 = ((28-conv1_size+2*2)/1+1)/2
        w_in_fc1 = ((28-conv1_size+2*2)/1+1)/2
        # Weight of FC1: 1176x120
        self.params['W3'] = weight_scale * np.random.randn(num_conv1 * h_in_fc1 * w_in_fc1, dim_fc1)
        self.params['b3'] = np.zeros(dim_fc1)

        # Weight of FC2: 120x84
        self.params['W4'] = weight_scale * np.random.randn(dim_fc1, dim_fc2)
        self.params['b4'] = np.zeros(dim_fc2)

        # Weight of FC3: 84x10
        self.params['W5'] = weight_scale * np.random.randn(dim_fc2, num_classes)
        self.params['b5'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        W1, b1 = self.params['W1'], self.params['b1']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        conv1_size = W1.shape[2]
        conv1_param = {'stride': 1, 'pad': 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # CONV1-RELU-POOLING: Input=Nx1x28x28, Output = Nx6x14x14
        A1, conv1_cache = conv_relu_pool_forward(X, W1, b1, conv1_param, pool_param)

        # FC1-RELU: Input=Nx16x5x5, Output = Nx120
        A3, fc1_cache = affine_relu_forward(A1, W3, b3)

        # FC2-RELU: Input=Nx120, Output = Nx84
        A4, fc2_cache = affine_relu_forward(A3, W4, b4)

        # FC3: Input=Nx84, Output = Nx10
        scores, fc3_cache = affine_forward(A4, W5, b5)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # Input = Nx1, Output = Nx10
        loss, dloss = cross_entropy(scores, y)

        # FC3 backward: Input=Nx10, Output = Nx84
        dfc3, grads['W5'], grads['b5'] = affine_backward(dloss, fc3_cache)

        # FC2-RELU backward: Input=Nx84, Output = Nx120
        dfc2, grads['W4'], grads['b4'] = affine_relu_backward(dfc3, fc2_cache)

        # FC1-RELU backward: Input=Nx120, Output = Nx16x5x5
        dfc1, grads['W3'], grads['b3'] = affine_relu_backward(dfc2, fc1_cache)

        # CONV1-RELU-POOLING backward: Input=Nx6x14x14, Output = Nx1x28x28
        dconv1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dfc1, conv1_cache)

        # Add L2 regularization
        # loss += 0.5 * self.reg * (
        # np.square(W1).sum() + np.square(W2).sum() + np.square(W3).sum() + np.square(W4).sum() + np.square(W5).sum())

        # grads['W1'] += self.reg * self.params['W1']
        # grads['W2'] += self.reg * self.params['W2']
        # grads['W3'] += self.reg * self.params['W3']
        # grads['W4'] += self.reg * self.params['W4']
        # grads['W5'] += self.reg * self.params['W5']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads