.. _mlp:

Multi-Layer Perceptron
----------------------

In this example we'll learn to use ``mlx.nn`` by implementing a simple
multi-layer perceptron to classify MNIST.

As a first step import the MLX packages we need:

.. code-block:: ruby

  require "mlx"
  mx = MLX::Core
  nn = MLX::NN
  optim = MLX::Optimizers
  # Ruby example: replace NumPy usage with a Ruby array library (e.g., numo-narray)

The model is defined as the ``MLP`` class which inherits from
:class:`mlx.nn.Module`. We follow the standard idiom to make a new module:

1. Define an ``initialize`` method where the parameters and/or submodules are setup. See
   the :ref:`Module class docs<module_class>` for more information on how
   :class:`mlx.nn.Module` registers parameters.
2. Define a ``call`` method where the computation is implemented.

.. code-block:: ruby

  class MLP < MLX::NN::Module
    def initialize(num_layers, input_dim, hidden_dim, output_dim)
      super()
      layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
      @layers = []
      (0...(layer_sizes.length - 1)).each do |i|
        @layers << MLX::NN::Linear.new(layer_sizes[i], layer_sizes[i + 1])
      end
    end

    def call(x)
      @layers[0...-1].each do |l|
        x = MLX::Core.maximum(l.call(x), 0.0)
      end
      @layers[-1].call(x)
    end
  end


We define the loss function which takes the mean of the per-example cross
entropy loss.  The ``mlx.nn.losses`` sub-package has implementations of some
commonly used loss functions.

.. code-block:: ruby

  def loss_fn(model, x, y)
      mx.mean(nn.losses.cross_entropy(model.call(x), y))
  end

We also need a function to compute the accuracy of the model on the validation
set:

.. code-block:: ruby

  def eval_fn(model, x, y)
      mx.mean(mx.argmax(model.call(x), axis: 1) == y)
  end

Next, setup the problem parameters and load the data. To load the data, you need our
`MNIST data loader
<https://github.com/ml-explore/mlx-examples/tree/main/mnist>`_, which should be
invoked from a Ruby script in your workflow.

.. code-block:: ruby

  num_layers = 2
  hidden_dim = 32
  num_classes = 10
  batch_size = 256
  num_epochs = 10
  learning_rate = 1e-1

  # Load the data with your preferred Ruby dataset helper
  # (for example, a Ruby MNIST loader gem):
  # train_images, train_labels, test_images, test_labels = mnist_data

Since we're using SGD, we need an iterator which shuffles and constructs
minibatches of examples in the training set:

.. code-block:: ruby

  def batch_iterate(batch_size, x, y)
      # Replace this with your preferred random-shuffle helper
      perm = (0...y.shape[0]).to_a.shuffle
      # Example assumes `perm` is used to index the mini-batches
      (0...y.shape[0]).step(batch_size).each do |s|
          ids = perm[s...(s + batch_size)]
          yield x[ids], y[ids]
      end
  end


Finally, we put it all together by instantiating the model, the
:class:`mlx.optimizers.SGD` optimizer, and running the training loop:

.. code-block:: ruby

  # Load the model
  model = MLP.new(num_layers, train_images.shape[-1], hidden_dim, num_classes)
  mx.eval(model.parameters)

  # Get a function which gives the loss and gradient of the
  # loss with respect to the model's trainable parameters
  loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

  # Instantiate the optimizer
  optimizer = optim::SGD.new(learning_rate: learning_rate)

  num_epochs.times do |e|
      batch_iterate(batch_size, train_images, train_labels).each do |x, y|
          loss, grads = loss_and_grad_fn.call(model, x, y)

          # Update the optimizer state and model parameters
          # in a single call
          optimizer.update(model, grads)

          # Force a graph evaluation
          mx.eval(model.parameters, optimizer.state)
      end

      accuracy = eval_fn(model, test_images, test_labels)
      puts "Epoch #{e}: Test accuracy #{accuracy.item.round(3)}"
  end

.. note::
  The :func:`mlx.nn.value_and_grad` function is a convenience function to get
  the gradient of a loss with respect to the trainable parameters of a model.
  This should not be confused with :func:`mlx.core.value_and_grad`.

The model should train to a decent accuracy (about 95%) after just a few passes
over the training set. The `full example <https://github.com/ml-explore/mlx-examples/tree/main/mnist>`_
is available in the MLX GitHub repo.
