import numpy as np
from scipy import signal
from convenient_math import *


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    # Get dimension of input x
    N = x.shape[0]

    # Get dimension of input w
    D = w.shape[0]
    M = w.shape[1]

    out = np.zeros((N, M), int)

    if (np.size(x)/N) != D or b.shape[0] != M:
        print('Wrong dimension, check dimension of x, w, b')
        quit()

    x_vec = x.reshape(N,D)
    #b = np.tile(b, (N,1)) # repeat vector b of N times
    out = np.dot(x_vec, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

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
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
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
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height h_conv and width h_conv.

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
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # Check dimension
    if w.shape[0] != b.shape[0]:
        print 'Wrong dimension of depth slice !'
        quit()

    # Get dimension of input x
    N    = x.shape[0]   # No. of data point in mini-batch
    c_in = x.shape[1]   # No. of channels in each data point
    h_in = x.shape[2]
    w_in = x.shape[3]
    # N, c_in, h_in, w_in = x.shape

    # Filter parameters
    pad    = conv_param['pad']
    stride = conv_param['stride']

    # Get dimension of filter w
    f_conv = w.shape[0] # No. of filters
    c_conv = w.shape[1] # No. of channels of filter
    h_conv = w.shape[2]
    w_conv = w.shape[3]

    # Compute dimension of output
    H_out = 1 + (h_in + 2 * pad - h_conv) / stride
    W_out = 1 + (w_in + 2 * pad - w_conv) / stride

    if (h_in + 2 * pad - h_conv) % stride != 0 or (w_in + 2 * pad - w_conv) % stride != 0:
        print 'Convolution parameters not valid'
        quit()
    else:
        out = np.zeros((N, f_conv, H_out, W_out))

    # Padding zeros to input
    if pad == 0:
        input_zeros_pad = x
    else:
        input_zeros_pad = np.zeros([N, c_in, h_in + 2 * pad, w_in + 2 * pad])
        input_zeros_pad[:, :, pad:pad + h_in, pad:pad + w_in] = x

    for n in range(N):  # No. of point (mini-batch)
        for f in range(f_conv): # No. of CONV
            # flt = np.flip(w, 2)
            # flt = np.flip(flt,3)
            for i in range(H_out):
                for j in range(W_out):
                    out[n, f, i, j] += np.sum(
                        input_zeros_pad[n, :, i*stride:i*stride + h_conv, j*stride:j*stride + w_conv]*w[f]) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    (x, w, b, conv_param) = cache
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # Get dimension of input
    N, f_in, h_in, w_in = dout.shape

    # Get dimension of filter
    fW, cW, hW, wW = w.shape

    # Get dimension of input x
    _, cX, hX, wX = x.shape

    # Filter parameters
    pad = conv_param['pad']
    stride = conv_param['stride']

    if f_in != fW:
        print 'Wrong dimension in previous propagation'
        quit()

    # Padding zeros to input
    input_zeros_pad = np.zeros([N, cX, hX + 2 * pad, wX + 2 * pad])
    input_zeros_pad[:, :, pad:pad + hX, pad:pad + wX] = x

    dw = np.zeros(w.shape, float)
    dx = np.zeros(input_zeros_pad.shape, float)
    db = np.zeros(b.shape, float)

    for n in range(N):
        for f in range(fW): # Iterate through all of filters
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
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.select([x > 0], [x])
    cache = x
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.zeros(x.shape, float)
    dx = np.select([cache > 0], [dout])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def sigmoid_forward(x):
    """
        Computes the forward pass for a layer of sigmoid function.

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
    out = None
    ###########################################################################
    # TODO: Implement the Sigmoid forward pass.                               #
    ###########################################################################
    out = sigmoid(x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache

def sigmoid_backward(dout, cache):
    """
        Computes the backward pass for a layer of sigmoid function.

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the sigmoid backward pass.                              #
    ###########################################################################
    dx_tmp = 1 - sigmoid(x)
    dx_tmp = np.multiply(dx_tmp, sigmoid(x))
    dx = np.multiply(dx_tmp, dout)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def tanh_forward(x):
    """
        Computes the forward pass for a layer of tanh function.

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
    out = None
    ###########################################################################
    # TODO: Implement the Tanh forward pass.                               #
    ###########################################################################
    out = tanh(x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache

def tanh_backward(dout, cache):
    """
        Computes the backward pass for a layer of tanh function.

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the Tanh backward pass.                              #
    ###########################################################################
    dx_tmp = np.multiply(tanh(x), tanh(x))
    dx_tmp = 1 - dx_tmp
    dx = np.multiply(dx_tmp, dout)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def max_pool_forward_naive(x, pool_param):
    # LUU Y: Ham chi ho tro stride khong bi overlap filter
    """
    A naive implementation of the forward pass for a max pooling layer.

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
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    # Get pooling parameter
    pool_heigh = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    # Get dimension of input
    N    = x.shape[0]
    cX = x.shape[1]
    hX = x.shape[2]
    wX = x.shape[3]

    # Get dimension of output
    h_out = (hX - pool_heigh)/stride + 1
    w_out = (wX - pool_width)/stride + 1

    x_tmp = np.zeros(x.shape,int)
    out = np.zeros((N, cX, h_out,w_out),float)

    for n in range(N):
        for c in range(cX):
            for i in range(h_out):
                for j in range(w_out):
                    out[n, c, i, j] = np.max(x[n, c, i*stride:i*stride+pool_heigh, j*stride:j*stride+pool_width])

                    # Create matrix saving coordinate of maximum element
                    id_max = np.argmax(x[n, c, i*stride:i*stride+pool_heigh, j*stride:j*stride+pool_width])
                    patch_stretch = np.zeros((1, pool_heigh*pool_width), int)
                    patch_stretch[0, id_max] = 1
                    patch_stretch = patch_stretch.reshape(pool_heigh, pool_width)
                    x_tmp[n, c, i * stride:i * stride + pool_heigh, j * stride:j * stride + pool_width] = patch_stretch
    x = x_tmp
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    (x, pool_param) = cache
    dx = np.zeros(x.shape, float)

    # Get dimension of input x
    N    = x.shape[0]
    c_in = x.shape[1]
    h_in = x.shape[2]
    w_in = x.shape[3]

    pool_heigh = pool_param['pool_height']
    pool_width = pool_param['pool_width']

    # Expand dout matrix
    dout = dout.repeat(pool_heigh, axis=2).repeat(pool_width, axis=3)


    for n in range(N):
        dx[n, :, :, :] = np.multiply(x[n,:,:,:], dout[n,:,:,:])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

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
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p)
        out = (x*mask)/p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = (dout*mask)/dropout_param['p']
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
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
    """

    N = X.shape[0]
    dx = softmax(X)
    log_likelihood = -np.log(dx[range(N),ground_truth])
    loss = np.sum(log_likelihood)/N
    dx[range(N), ground_truth] -= 1
    dx = dx/N

    return loss, dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx