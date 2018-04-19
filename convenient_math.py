import numpy as np

def sigmoid(x):
    """
        Implement of sigmoid function, can be applied element-wise

        Input:
            x: Input, of any shape
        Return:
            Value of sigmoid function applied with each element in x
    """
    return 1/(1+np.exp(-x))

def tanh(x):
    """
        Implement of tanh function, can be applied element-wise

        Input:
            x: Input, of any shape
        Return:
            Value of tanh function applied with each element in x
    """
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def softmax(x):
    """
        Implement of softmax function, can be applied element-wise

        Input:
            x: Input, of any shape
        Return:
            Value of softmax function applied with each element in x
    """
    out = np.zeros(x.shape,float)
    e_x = np.zeros(x.shape,float)
    for i in range(x.shape[0]):
        e_x[i,:] = np.exp(x[i,:] - np.max(x[i,:])) # Help numerical stability
        out[i,:] = e_x[i,:]/np.sum(e_x[i,:])
    return out

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))