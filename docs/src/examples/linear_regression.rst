.. _linear_regression:

Linear Regression
-----------------

Let's implement a basic linear regression model as a starting point to
learn MLX. First import the core package and setup some problem metadata:

.. code-block:: ruby

  require "mlx"
  mx = MLX::Core
  num_features = 100
  num_examples = 1_000
  num_iters = 200  # iterations of SGD
  lr = 0.01  # learning rate for SGD


We'll generate a synthetic dataset by:

1. Sampling the design matrix ``X``.
2. Sampling a ground truth parameter vector ``w_star``.
3. Compute the dependent values ``y`` by adding Gaussian noise to ``X @ w_star``.

.. code-block:: ruby

  # Ground-truth parameters
  w_star = mx.random_uniform([num_features], -1.0, 1.0, mx.float32)

  # Input examples (design matrix)
  x_train = mx.random_uniform([num_examples, num_features], -1.0, 1.0, mx.float32)

  # Noisy labels
  eps = mx.random_uniform([num_examples], -1.0, 1.0, mx.float32) * 1e-2
  y = mx.matmul(x_train, w_star) + eps


We will use SGD to find the optimal weights. To start, define the squared loss
and get the gradient function of the loss with respect to the parameters.

.. code-block:: ruby

  loss_fn = ->(w) do
    mx.mean(mx.square(mx.matmul(x_train, w) - y)) * 0.5
  end

  grad_fn = mx.grad(loss_fn)

Start the optimization by initializing the parameters ``w`` randomly. Then
repeatedly update the parameters for ``num_iters`` iterations. 

.. code-block:: ruby

  w = mx.random_uniform([num_features], -1.0, 1.0, mx.float32) * 1e-2

  num_iters.times do
    grad = grad_fn.call(w)
    w = w - grad * lr
    mx.eval(w)
  end

Finally, compute the loss of the learned parameters and verify that they are
close to the ground truth parameters.

.. code-block:: ruby

  loss = mx.mean(mx.square(mx.matmul(x_train, w) - y)) * 0.5
  error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5

  puts format("Loss %.5f, |w-w*| = %.5f", loss.item, error_norm)
  # Should print something close to: Loss 0.00005, |w-w*| = 0.00364

Full versions of the linear and logistic regression examples are available in the
MLX examples repository under the Ruby-focused examples for supervised learning.
