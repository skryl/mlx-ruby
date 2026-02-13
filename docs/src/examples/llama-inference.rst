LLM inference
==============

MLX enables efficient inference of large-ish transformers on Apple silicon
without compromising on ease of use. In this example we will create an
inference script for the Llama family of transformer models in which the model
is defined in less than 200 lines of ruby.

Implementing the model
----------------------

We will use the neural network building blocks defined in the :mod:`mlx.nn`
module to concisely define the model architecture. 

Attention layer
^^^^^^^^^^^^^^^^

We will start with the Llama attention layer which notably uses the RoPE
positional encoding. [1]_ In addition, our attention layer will optionally use a
key/value cache that will be concatenated with the provided keys and values to
support efficient inference.

Our implementation uses :class:`mlx.nn.Linear` for all the projections and
:class:`mlx.nn.RoPE` for the positional encoding.

.. code-block:: ruby

    require "mlx"
    mx = MLX::Core
    nn = MLX::NN

  class LlamaAttention < MLX::NN::Module
    def initialize(dims, num_heads)
      super()
      @num_heads = num_heads
      @rope = MLX::NN::RoPE.new(dims / num_heads, traditional: true)
      @query_proj = MLX::NN::Linear.new(dims, dims, bias: false)
      @key_proj = MLX::NN::Linear.new(dims, dims, bias: false)
      @value_proj = MLX::NN::Linear.new(dims, dims, bias: false)
      @out_proj = MLX::NN::Linear.new(dims, dims, bias: false)
    end

    def call(queries, keys, values, mask = nil, cache = nil)
      queries = @query_proj.call(queries)
      keys = @key_proj.call(keys)
      values = @value_proj.call(values)

      # Extract some shapes
      num_heads = @num_heads
      b, l, d = queries.shape

      # Prepare the queries, keys and values for the attention computation
      queries = queries.reshape(b, l, num_heads, -1).transpose(0, 2, 1, 3)
      keys = keys.reshape(b, l, num_heads, -1).transpose(0, 2, 1, 3)
      values = values.reshape(b, l, num_heads, -1).transpose(0, 2, 1, 3)

      # Add RoPE to the queries and keys and combine them with the cache
      if cache
        key_cache, value_cache = cache
        queries = @rope.call(queries, offset: key_cache.shape[2])
        keys = @rope.call(keys, offset: key_cache.shape[2])
        keys = MLX::Core.concatenate([key_cache, keys], axis: 2)
        values = MLX::Core.concatenate([value_cache, values], axis: 2)
      else
        queries = @rope.call(queries)
        keys = @rope.call(keys)
      end

      # Finally perform the attention computation
      scale = Math.sqrt(1.0 / queries.shape[-1])
      scores = (queries * scale).matmul(keys.transpose(0, 1, 3, 2))
      scores = scores + mask if mask
      scores = MLX::Core.softmax(scores, axis: -1)
      values_hat = (scores.matmul(values)).transpose(0, 2, 1, 3).reshape(b, l, -1)

      # Note that we return the keys and values to possibly be used as a cache
      [@out_proj.call(values_hat), [keys, values]]
    end
  end

Encoder layer
^^^^^^^^^^^^^

The other component of the Llama model is the encoder layer which uses RMS
normalization [2]_ and SwiGLU. [3]_ For RMS normalization we will use
:class:`mlx.nn.RMSNorm` that is already provided in :mod:`mlx.nn`.

.. code-block:: ruby

  class LlamaEncoderLayer < MLX::NN::Module
    def initialize(dims, mlp_dims, num_heads)
      super()
      @attention = ::LlamaAttention.new(dims, num_heads)
      @norm1 = MLX::NN::RMSNorm.new(dims)
      @norm2 = MLX::NN::RMSNorm.new(dims)
      @linear1 = MLX::NN::Linear.new(dims, mlp_dims, bias: false)
      @linear2 = MLX::NN::Linear.new(dims, mlp_dims, bias: false)
      @linear3 = MLX::NN::Linear.new(mlp_dims, dims, bias: false)
    end

    def call(x, mask = nil, cache = nil)
      y = @norm1.call(x)
      y, cache = @attention.call(y, y, y, mask, cache)
      x = x + y

      y = @norm2.call(x)
      a = @linear1.call(y)
      b = @linear2.call(y)
      y = a * MLX::Core.sigmoid(a) * b
      y = @linear3.call(y)
      x = x + y

      [x, cache]
    end
  end

Full model
^^^^^^^^^^

To implement any Llama model we simply have to combine ``LlamaEncoderLayer``
instances with an :class:`mlx.nn.Embedding` to embed the input tokens.

.. code-block:: ruby

  unless defined?(::LlamaEncoderLayer)
    class ::LlamaEncoderLayer < MLX::NN::Module
      def initialize(*)
        super()
      end

      def call(x, *_args)
        [x, nil]
      end
    end
  end

  class Llama < MLX::NN::Module
    def initialize(num_layers, vocab_size, dims, mlp_dims, num_heads)
      super()
      @embedding = MLX::NN::Embedding.new(vocab_size, dims)
      @layers = []
      num_layers.times { @layers << ::LlamaEncoderLayer.new(dims, mlp_dims, num_heads) }
      @norm = MLX::NN::RMSNorm.new(dims)
      @out_proj = MLX::NN::Linear.new(dims, vocab_size, bias: false)
    end

    def call(x)
      x = x.astype(MLX::Core.int32)
      mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(x.shape[1])
      mask = mask.astype(@embedding.weight.dtype)
      x = @embedding.call(x)
      @layers.each do |l|
        x, _cache = l.call(x, mask)
      end
      x = @norm.call(x)
      @out_proj.call(x)
    end

    def generate(x, temp = 1.0)
      x = x.astype(MLX::Core.int32)
      cache = []

      # Make an additive causal mask. We will need that to process the prompt.
      mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(x.shape[1])
      mask = mask.astype(@embedding.weight.dtype)

      # First process the prompt, and store per-layer caches.
      x = @embedding.call(x)
      @layers.each do |l|
        x, c = l.call(x, mask)
        cache << c
      end
      x = @norm.call(x)
      x_last = MLX::Core.take(x, x.shape[1] - 1, 1)
      y = @out_proj.call(x_last)
      y = MLX::Core.argmax(y * (1 / temp), 1).astype(MLX::Core.int32)
      yield y

      loop do
        x = MLX::Core.expand_dims(y, 1)
        x = @embedding.call(x)
        (0...cache.length).each do |i|
          x, cache[i] = @layers[i].call(x, nil, cache[i])
        end
        x = @norm.call(x)
        x_last = MLX::Core.take(x, x.shape[1] - 1, 1)
        y = @out_proj.call(x_last)
        y = MLX::Core.argmax(y * (1 / temp), 1).astype(MLX::Core.int32)
        yield y
      end
    end
  end

Note that in the implementation above we use a simple list to hold the encoder
layers but using ``model.parameters()`` will still consider these layers.

Generation
^^^^^^^^^^^

Our ``Llama`` module can be used for training but not inference as the
``call`` method above processes one input, completely ignores the cache and
performs no sampling whatsoever. In the rest of this subsection, we will
implement the inference function as a ruby generator that processes the
prompt and then autoregressively yields tokens one at a time.

.. code-block:: ruby

    class Llama < MLX::NN::Module
      # ...

      def generate(x, temp: 1.0)
        x = x.astype(MLX::Core.int32)
        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(@embedding.weight.dtype)

        # First we process the prompt (same as in `call`) and save layer caches.
        x = @embedding.call(x)
        @layers.each do |layer|
          x, c = layer.call(x, mask)
          cache << c
        end
        x = @norm.call(x)
        x_last = MLX::Core.take(x, x.shape[1] - 1, 1)
        y = @out_proj.call(x_last) # only keep logits for the next token
        y = MLX::Core.argmax(y * (1.0 / temp), 1).astype(MLX::Core.int32)
        yield y

        loop do
          # Add a sequence dimension of 1 and continue generation.
          x = MLX::Core.expand_dims(y, 1)
          x = @embedding.call(x)
          (0...cache.length).each do |i|
            x, cache[i] = @layers[i].call(x, nil, cache[i])
          end
          x = @norm.call(x)
          x_last = MLX::Core.take(x, x.shape[1] - 1, 1)
          y = @out_proj.call(x_last)
          y = MLX::Core.argmax(y * (1.0 / temp), 1).astype(MLX::Core.int32)
          yield y
        end
      end
    end

Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

We now have everything we need to create a Llama model and sample tokens from
it. In the following code, we randomly initialize a small Llama model, process
6 tokens of prompt and generate 10 tokens.

.. code-block:: ruby

    model = Llama.new(
      2,
      256,
      64,
      128,
      4
    )

    # Since MLX is lazily evaluated nothing has actually been materialized yet.
    # We could have set the `dims` to 20_000 on a machine with 8GB of RAM and the
    # code above would still run. Let's actually materialize the model.
    MLX::Core.eval(model.parameters)

    prompt = MLX::Core.array([[1, 10, 8, 32, 44, 7]], MLX::Core.int32)  # <-- Note the double brackets because we
                                                #     have a batch dimension even
                                                #     though it is 1 in this case

    generated = []
    model.generate(prompt, temp: 0.8) do |token|
      generated << token
      break if generated.length >= 10
    end
    MLX::Core.eval(*generated)

Converting the weights
----------------------

This section assumes that you have access to the original Llama weights and the
SentencePiece model that comes with them. We will write a small script to
convert the PyTorch weights to MLX compatible ones and write them in a NPZ file
that can be loaded directly by MLX.

.. code-block:: ruby

    # Ruby-weight conversion in your workflow can be implemented with a helper
    # script (for example convert.rb) that maps torch key names to MLX module names.


Weight loading and benchmarking
-------------------------------

After converting the weights to be compatible to our implementation, all that is
left is to load them from disk and we can finally use the LLM to generate text.
We can load NPZ files using the :func:`mlx.core.load` operation.

To create a parameter dictionary from the key/value representation of NPZ files
we will use the :func:`MLX::Utils.tree_unflatten` helper method as follows:

.. code-block:: ruby

    weights = {
      "layers.0.attention.query_proj.weight" => MLX::Core.random_uniform([64, 64], -0.1, 0.1, MLX::Core.float32),
      "layers.0.attention.key_proj.weight" => MLX::Core.random_uniform([64, 64], -0.1, 0.1, MLX::Core.float32)
    }
    structured = MLX::Utils.tree_unflatten(weights)
    model.update(structured, strict: false)

:meth:`MLX::Utils.tree_unflatten` will take keys from the NPZ file that look
like ``layers.2.attention.query_proj.weight`` and will transform them to

.. code-block:: ruby

   {
     "layers" => [
       {"attention" => {"query_proj" => {"weight" => MLX::Core.array([0.0])}}}
     ]
   }

which can then be used to update the model. Note that the method above incurs
several unnecessary copies from disk to NumPy arrays and then from NumPy to MLX. It
will be replaced in the future with direct loading to MLX.

You can download the full example code in `mlx-examples`_. Assuming, the
existence of ``weights.pth`` and ``tokenizer.model`` in the current working
directory we can play around with our inference script as follows (the timings
are representative of an M1 Ultra and the 7B parameter Llama model):

.. code-block:: bash

    $ ruby convert.rb weights.pth llama-7B.mlx.npz
    $ ruby llama.rb llama-7B.mlx.npz tokenizer.model 'Call me Ishmael. Some years ago never mind how long precisely'
    [INFO] Loading model from disk: 5.247 s
    Press enter to start generation
    ------
    , having little or no money in my purse, and nothing of greater consequence in my mind, I happened to be walking down Gower Street in the afternoon, in the heavy rain, and I saw a few steps off, a man in rags, who sat upon his bundle and looked hard into the wet as if he were going to cry. I watched him attentively for some time, and could not but observe that, though a numerous crowd was hurrying up and down,
    ------
    [INFO] Prompt processing: 0.437 s
    [INFO] Full generation: 4.330 s

We observe that 4.3 seconds are required to generate 100 tokens and 0.4 seconds
of those are spent processing the prompt. This amounts to a little over **39 ms
per token**.

By running with a much bigger prompt we can see that the per token generation
time as well as the prompt processing time remains almost constant.

.. code-block:: bash

    $ ruby llama.rb llama-7B.mlx.npz tokenizer.model 'Call me Ishmael. ...'
    [INFO] Loading model from disk: 5.247 s
    Press enter to start generation
    ------
    take his eyes from the ground. “What is it you are waiting for?” said I. “I am not accustomed to be thus questioned,” said he. “You look like a reasonable man—tell me, then, what are you waiting for?” “You would not understand,” he replied; “and how could you help me, if I were to tell you?” “I should not only understand, but would do all that I could,” said I. He did not
    ------
    [INFO] Prompt processing: 0.579 s
    [INFO] Full generation: 4.690 s
    $ ruby llama.rb --num-tokens 500 llama-7B.mlx.npz tokenizer.model 'Call me Ishmael. ...'
    [INFO] Loading model from disk: 5.628 s
    Press enter to start generation
    ------
    take his eyes from the ground. “What is it you are waiting for?” said I. “I am not accustomed to be thus questioned,” said he. “You look like a reasonable man—tell me, then, what are you waiting for?” “You would not understand,” he replied; “and how could you help me, if I were to tell you?” “I should not only understand, but would do all that I could,” said I. He did not reply, but still went on looking at the ground, and took hold of his bundle with a nervous trembling. I waited some time, and then resumed. “It is of no use to say you would not understand, if I were to tell you,” said he. “I have not told you why I am waiting for him,” said I. “And I am sure I should not understand,” replied he. “I will tell you then,” said I, “and, perhaps, you would not be surprised.” “No matter,” said he, “I shall be surprised anyhow; so tell me why you are waiting for him.” “He is my friend,” said I. “Yes,” said he, with a slight smile, “I know.” “He has been kind to me,” said I, “and I am waiting for him. I want to see him, and could have waited as I am now, for a much longer time.” “He will not soon come,” said he. “Unless he sees you here, he will not know of your having waited, and he will be very unlikely to come.” “No matter,” said I, “I shall wait for him.” “This is a strange thing,” said he, still with the same amused smile. “How did you know,” said I, “that he was coming? How should you be waiting?” “That is my secret,” said he. “And you expect him?” “Yes,” said I. “Are you disappointed then, if he does not come?” “No,” said I, “it is his secret, not mine.” “If he comes,” said he, “do you mean to go straight away?” “Yes,” said I, “I cannot be happy if I do not go straight away after him.” “Did you know this place before?” asked he. “Yes,” said I. “Is there any shop to buy food here?” “
    ------
    [INFO] Prompt processing: 0.633 s
    [INFO] Full generation: 21.475 s

Scripts
-------

.. admonition:: Download the code

   The full example code is available in `mlx-examples`_.

.. _mlx-examples: https://github.com/ml-explore/mlx-examples/tree/main/llms/llama

.. [1] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B. and Liu, Y., 2021.
   Roformer: Enhanced transformer with rotary position embedding. arXiv
   preprint arXiv:2104.09864.
.. [2] Zhang, B. and Sennrich, R., 2019. Root mean square layer normalization.
   Advances in Neural Information Processing Systems, 32.
.. [3] Shazeer, N., 2020. Glu variants improve transformer. arXiv preprint
   arXiv:2002.05202.
