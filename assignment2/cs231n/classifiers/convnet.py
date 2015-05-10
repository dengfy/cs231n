import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def two_layer_convnet(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradient for a simple two-layer ConvNet. The architecture
  is conv-relu-pool-affine-softmax, where the conv layer uses stride-1 "same"
  convolutions to preserve the input size; the pool layer uses non-overlapping
  2x2 pooling regions. We use L2 regularization on both the convolutional layer
  weights and the affine layer weights.

  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - model: Dictionary mapping parameter names to parameters. A two-layer Convnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the affine layer
  - y: Vector of labels of shape (N,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, C, H, W = X.shape

  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  scores, cache2 = affine_forward(a1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW2, db2 = affine_backward(dscores, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  
  return loss, grads


def init_two_layer_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  """
  Initialize the weights for a two-layer ConvNet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_shape: Tuple giving the input shape to the network; default is
    (3, 32, 32) for CIFAR-10.
  - num_classes: The number of classes for this network. Default is 10
    (for CIFAR-10)
  - num_filters: The number of filters to use in the convolutional layer.
  - filter_size: The width and height for convolutional filters. We assume that
    all convolutions are "same", so we pick padding to ensure that data has the
    same height and width after convolution. This means that the filter size
    must be odd.

  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the fully-connected layer.
  """
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, num_classes)
  model['b2'] = bias_scale * np.random.randn(num_classes)
  return model


def init_supercool_convnet(weight_scale=1e-5, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward 
  C, H, W = input_shape

  model = {}
  model['W1'] = weight_scale * np.random.randn(16, 3, 5, 5)
  model['W2'] = weight_scale * np.random.randn(20, 16, 5, 5)
  model['W3'] = weight_scale * np.random.randn(20, 20, 5, 5)
  model['W4'] = weight_scale * np.random.randn(20 * 4 * 4, num_classes)

  model['b1'] = bias_scale * np.random.randn(16)
  model['b2'] = bias_scale * np.random.randn(20)
  model['b3'] = bias_scale * np.random.randn(20)
  model['b4'] = bias_scale * np.random.randn(num_classes)
  return model


def supercool_convnet(X, model, y=None, reg=0.0):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward 
  W1 = model['W1']
  W2 = model['W2']
  W3 = model['W3']
  W4 = model['W4']

  b1 = model['b1']
  b2 = model['b2']
  b3 = model['b3']
  b4 = model['b4']


  N, C, H, W = X.shape
  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  # Forward
  conv_param = {'stride': 1, 'pad': 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  out2, cache2 = conv_relu_pool_forward(out1, W2, b2, conv_param, pool_param)
  out3, cache3 = conv_relu_pool_forward(out2, W3, b3, conv_param, pool_param)
  scores, cache4 = affine_forward(out3, W4, b4)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  dout3, dW4, db4 = affine_backward(dscores, cache4)
  dout2,  dW3, db3 = conv_relu_pool_backward(dout3, cache3)
  dout1,  dW2, db2 = conv_relu_pool_backward(dout2, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(dout1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'W2': dW2, 'W3': dW3, 'W4': dW4, 'b1': db1, 'b2': db2, 'b3': db3, 'b4': db4}
  
  return loss, grads