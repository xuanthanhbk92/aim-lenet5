import numpy as np
from scipy import signal
from convenient_math import *


def affine_forward(x, w, b):
    """
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    # Get dimension of input x
    N = x.shape[0]

    # Get dimension of input w
    D = w.shape[0]
    M = w.shape[1]

    out = np.zeros((N, M))
    if (np.size(x)/N) != D or b.shape[0] != M:
        print('Wrong dimension, check dimension of x, w, b')
        quit()

    x_vec = x.reshape(N,D)
    out = np.dot(x_vec, w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Bias, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    # Get shape of input w and dout
    D = w.shape[0]
    N = dout.shape[0]
    M = dout.shape[1]

    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    x_vec = x.reshape(N, D) # Stretch vector of x

    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)

    dw = np.dot(x_vec.T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db

def conv_forward_naive(x, w, b, conv_param):
    """
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, h_conv, w_conv)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - h_conv) / stride
      W' = 1 + (W + 2 * pad - w_conv) / stride
    - cache: (x, w, b, conv_param)
    """

    # Check dimension
    if w.shape[0] != b.shape[0]:
        print 'Wrong dimension of depth slice !'
        quit()

    # Get dimension of input x
    N, c_in, h_in, w_in = x.shape

    # Filter parameters
    pad, stride = conv_param['pad'], conv_param['stride']

    # Get dimension of filter w
    f_conv, c_conv, h_conv, w_conv = w.shape

    # Compute dimension of output
    H_out = 1 + (h_in + 2 * pad - h_conv) / stride
    W_out = 1 + (w_in + 2 * pad - w_conv) / stride

    if (h_in + 2 * pad - h_conv) % stride != 0 or (w_in + 2 * pad - w_conv) % stride != 0:
        print 'Convolution parameters not valid'
        quit()

    out = np.zeros((N, f_conv, H_out, W_out))

    # Padding zeros to input
    input_zeros_pad = np.zeros([N, c_in, h_in + 2 * pad, w_in + 2 * pad])
    input_zeros_pad[:, :, pad:pad + h_in, pad:pad + w_in] = x

    for n in range(N):  # No. of point (mini-batch)
        for f in range(f_conv): # No. of CONV
            for i in range(H_out):
                for j in range(W_out):
                    out[n, f, i, j] += np.sum(
                        input_zeros_pad[n, :, i*stride:i*stride + h_conv, j*stride:j*stride + w_conv]*w[f]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    (x, w, b, conv_param) = cache

    # Get dimension of input
    N, f_in, h_in, w_in = dout.shape

    # Get dimension of filter
    fW, cW, hW, wW = w.shape

    # Get dimension of input x
    _, cX, hX, wX = x.shape

    # Filter parameters
    pad, stride = conv_param['pad'], conv_param['stride']

    if f_in != fW:
        print 'Wrong dimension in previous propagation'
        quit()

    # Padding zeros to input
    input_zeros_pad = np.zeros([N, cX, hX + 2 * pad, wX + 2 * pad])
    input_zeros_pad[:, :, pad:pad + hX, pad:pad + wX] = x

    dw = np.zeros(w.shape)
    dx = np.zeros(input_zeros_pad.shape)
    db = np.zeros(b.shape)

    for n in range(N):
        for f in range(fW):
            for i in range(h_in/stride):
                for j in range(w_in/stride):
                    db[f] += dout[n, f, i, j]
                    dw[f] += input_zeros_pad[n, :, i * stride:i * stride + hW, j*stride:j*stride + wW] * dout[n, f, i, j]
                    dx[n, :, i:i + hW, j:j + wW] += w[f] * dout[n,f ,i, j]

    dx = dx[:,:, pad:pad+hX, pad:pad+wX]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def relu_forward(x):
    """
    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """

    out = np.select([x > 0], [x])

    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache

    dx = np.zeros(x.shape)
    dx = np.select([x > 0], [dout])

    return dx

def sigmoid_forward(x):
    """
    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = sigmoid(x)

    cache = x
    return out, cache

def sigmoid_backward(dout, cache):
    """
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache

    tmp = 1 - sigmoid(x)
    tmp = np.multiply(tmp, sigmoid(x))

    dx = np.multiply(tmp, dout)

    return dx

def tanh_forward(x):
    """
    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """

    out = tanh(x)

    cache = x
    return out, cache

def tanh_backward(dout, cache):
    """
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache

    dx_tmp = np.multiply(tanh(x), tanh(x))
    dx_tmp = 1 - dx_tmp

    dx = np.multiply(dx_tmp, dout)

    return dx

def max_pool_forward_naive(x, pool_param):
    # Only suppport pooling not be overlap
    """
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """

    # Get pooling parameter
    pool_heigh , pool_width,stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    # Get dimension of input
    N, cX, hX, wX    = x.shape

    # Get dimension of output
    h_out = (hX - pool_heigh)/stride + 1
    w_out = (wX - pool_width)/stride + 1

    x_tmp = np.zeros(x.shape)
    out = np.zeros((N, cX, h_out,w_out))

    for n in range(N):
        for c in range(cX):
            for i in range(h_out):
                for j in range(w_out):
                    out[n, c, i, j] = np.max(x[n, c, i*stride:i*stride+pool_heigh, j*stride:j*stride+pool_width])

                    # Create matrix saving coordinate of maximum element
                    id_max = np.argmax(x[n, c, i*stride:i*stride+pool_heigh, j*stride:j*stride+pool_width])
                    patch_stretch = np.zeros((1, pool_heigh*pool_width))
                    patch_stretch[0, id_max] = 1
                    patch_stretch = patch_stretch.reshape(pool_heigh, pool_width)
                    x_tmp[n, c, i * stride:i * stride + pool_heigh, j * stride:j * stride + pool_width] = patch_stretch

    cache = (x_tmp, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    (x, pool_param) = cache

    (pool_heigh, pool_width) = pool_param['pool_height'], pool_param['pool_width']

    dx = np.zeros(x.shape)
    N, c_in, h_in, w_in = x.shape

    # Expand dout matrix
    dout = dout.repeat(pool_heigh, axis=2).repeat(pool_width, axis=3)

    for n in range(N):
        dx[n] = np.multiply(x[n], dout[n])

    return dx

def dropout_forward(x, dropout_param):
    """
    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p)
        out = (x*mask)/p
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def dropout_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = (dout*mask)/dropout_param['p']
    elif mode == 'test':
        dx = dout
    return dx

def cross_entropy(X, ground_truth):
    """
    Inputs:
        - X: of shape (N, M)
        - ground_truth: int indicating class, of shape (N,)
    Return:
        - loss: type of scalar
        - dloss: of shape (N, M)
    """

    N = X.shape[0]
    dx = softmax(X)

    log_likelihood = -np.log(dx[range(N),ground_truth])
    loss = np.sum(log_likelihood)/N

    dx[range(N), ground_truth] -= 1
    dx = dx/N

    return loss, dx

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta