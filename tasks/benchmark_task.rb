# frozen_string_literal: true

require "json"
require "open3"
require "fileutils"
LIB_ROOT = File.expand_path("../lib", __dir__)
$LOAD_PATH.unshift(LIB_ROOT) unless $LOAD_PATH.include?(LIB_ROOT)
require "mlx"

class BenchmarkTask
  CNN_CHANNELS = 3
  CNN_HEIGHT = 64
  CNN_WIDTH = 64
  CNN_CLASSES = 1024
  MLP_FACTOR = 4
  RNN_HIDDEN_MULTIPLIER = 2

  DEFAULT_ITERATIONS = 50
  DEFAULT_WARMUP = 10
  DEFAULT_BATCH_SIZE = 8
  DEFAULT_SEQUENCE_LENGTH = 128
  DEFAULT_TARGET_SEQUENCE_LENGTH = 64
  DEFAULT_DIMS = 256
  DEFAULT_HEADS = 8
  DEFAULT_LAYERS = 4
  DEFAULT_DTYPE = MLX::Core.float32

  class KarpathyGPT2Model < MLX::NN::Module
    def initialize(vocab_size:, dims:, num_heads:, num_layers:, block_size:, dropout: 0.0)
      super()

      self.token_embedding = MLX::NN::Embedding.new(vocab_size, dims)
      self.pos_embedding = MLX::NN::Embedding.new(block_size, dims)
      self.dropout = MLX::NN::Dropout.new(dropout)
      self.transformer_blocks = Array.new(num_layers) do
        MLX::NN::TransformerEncoderLayer.new(
          dims,
          num_heads,
          mlp_dims: dims * 4,
          dropout: dropout,
          norm_first: true
        )
      end
      self.layer_norm = MLX::NN::LayerNorm.new(dims)
      self.proj = MLX::NN::Linear.new(dims, vocab_size)
      @causal_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(block_size)
    end

    def call(input_ids)
      positions = MLX::Core.arange(0, input_ids.shape[1], 1, MLX::Core.int32)
      hidden = MLX::Core.add(
        token_embedding.call(input_ids),
        pos_embedding.call(positions)
      )
      hidden = dropout.call(hidden)

      transformer_blocks.each do |transformer_block|
        hidden = transformer_block.call(hidden, @causal_mask)
      end

      proj.call(layer_norm.call(hidden))
    end
  end

  def initialize(
    iterations: DEFAULT_ITERATIONS,
    warmup: DEFAULT_WARMUP,
    batch_size: DEFAULT_BATCH_SIZE,
    sequence_length: DEFAULT_SEQUENCE_LENGTH,
    target_sequence_length: DEFAULT_TARGET_SEQUENCE_LENGTH,
    dims: DEFAULT_DIMS,
    num_heads: DEFAULT_HEADS,
    num_layers: DEFAULT_LAYERS,
    compute_device: :gpu,
    python_bin: "python3"
  )
    @iterations = iterations
    @warmup = warmup
    @batch_size = batch_size
    @sequence_length = sequence_length
    @target_sequence_length = target_sequence_length
    @dims = dims
    @num_heads = num_heads
    @num_layers = num_layers
    @compute_device = parse_compute_device(compute_device)
    @python_bin = python_bin
    @repo_root = File.expand_path("..", __dir__)
  end

  def run(model: :transformer)
    model_name = model.to_sym
    raise "Unknown benchmark model: #{model_name}" unless available_models.include?(model_name)

    ruby_result = benchmark_ruby(model_name)
    python_result = benchmark_python(model_name)
    speedup = python_result.fetch("average_ms") / ruby_result.fetch("average_ms")

    puts "Benchmark (ruby vs python): #{model_name}"
    puts "  configuration: #{configuration_summary(model_name)}"
    puts "  compute device: #{compute_device_name}"
    puts "  iterations: #{@iterations}, warmup: #{@warmup}"
    puts "  ruby_avg_ms:   #{format('%.3f', ruby_result.fetch('average_ms'))}"
    puts "  python_avg_ms: #{format('%.3f', python_result.fetch('average_ms'))}"
    puts "  python/ruby:   #{format('%.2f', speedup)}x"
    puts "  output shape:  #{ruby_result.fetch('output_shape').join('x')} (ruby), " \
      "#{python_result.fetch('output_shape').join('x')} (python)"
    puts

    {
      "ruby" => ruby_result,
      "python" => python_result,
      "python_per_ruby" => speedup
    }
  end

  private

  def available_models
    [:transformer, :cnn, :mlp, :rnn, :karpathy_gpt2]
  end

  def configuration_summary(model_name)
    case model_name
    when :transformer
      "batch=#{@batch_size}, src_seq=#{@sequence_length}, tgt_seq=#{@target_sequence_length}, " \
        "dims=#{@dims}, heads=#{@num_heads}, layers=#{@num_layers}"
    when :cnn
      "batch=#{@batch_size}, channels=#{CNN_CHANNELS}, height=#{CNN_HEIGHT}, width=#{CNN_WIDTH}, classes=#{CNN_CLASSES}"
    when :mlp
      "batch=#{@batch_size}, input=#{mlp_input_size}, hidden=#{mlp_hidden_size}, output=#{mlp_output_size}"
    when :rnn
      "batch=#{@batch_size}, seq_len=#{@sequence_length}, input=#{@dims}, hidden=#{rnn_hidden_size}"
    when :karpathy_gpt2
      "batch=#{@batch_size}, block=#{@sequence_length}, dims=#{@dims}, heads=#{@num_heads}, layers=#{@num_layers}, vocab=<dataset>"
    else
      ""
    end
  end

  def compute_device_name
    @compute_device.to_s
  end

  def parse_compute_device(value)
    device = value.to_s.downcase.to_sym
    return :cpu if device == :cpu
    return :gpu if device == :metal || device == :gpu

    raise ArgumentError, "Unsupported compute device: #{value}. Use :cpu or :gpu."
  end

  def python_device_expr
    @compute_device == :cpu ? "mx.cpu" : "mx.gpu"
  end

  def mlp_input_size
    @dims * MLP_FACTOR
  end

  def mlp_hidden_size
    @dims * MLP_FACTOR
  end

  def mlp_output_size
    @dims
  end

  def rnn_hidden_size
    @dims * RNN_HIDDEN_MULTIPLIER
  end

  def benchmark_ruby(model_name)
    with_compute_device do
      case model_name
      when :transformer
        benchmark_ruby_transformer
      when :cnn
        benchmark_ruby_cnn
      when :mlp
        benchmark_ruby_mlp
      when :rnn
        benchmark_ruby_rnn
      when :karpathy_gpt2
        benchmark_ruby_karpathy_gpt2
      end
    end
  end

  def benchmark_ruby_karpathy_gpt2
    dataset = prepare_karpathy_gpt2_dataset
    train_data = dataset.fetch("train")
    vocab_size = dataset.fetch("vocab_size")

    model = KarpathyGPT2Model.new(
      vocab_size: vocab_size,
      dims: @dims,
      num_heads: @num_heads,
      num_layers: @num_layers,
      block_size: @sequence_length
    )
    value_grad = MLX::NN.value_and_grad(
      model,
      lambda do |input_batch, target_batch|
        logits = model.call(input_batch)
        logits_shape = logits.shape
        batch_size = logits_shape[0]
        sequence_length = logits_shape[1]
        vocab_size = logits_shape[2]

        reduced_logits = MLX::Core.reshape(
          MLX::Core.slice(logits, [0, 0, 0], [batch_size, sequence_length - 1, vocab_size]),
          [batch_size * (sequence_length - 1), vocab_size]
        )
        reduced_targets = MLX::Core.reshape(
          MLX::Core.slice(target_batch, [0, 1], [batch_size, sequence_length]),
          [batch_size * (sequence_length - 1)]
        )
        MLX::NN.cross_entropy(reduced_logits, reduced_targets, reduction: "mean")
      end
    )

    optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-3)
    rng = Random.new(0)
    output_shape = nil
    warmup_every = log_interval(@warmup)
    iter_every = log_interval(@iterations)

    sample_input, = next_karpathy_gpt2_batch(train_data, rng)
    output_shape = model.call(sample_input).shape

    @warmup.times do |idx|
      input_batch, target_batch = next_karpathy_gpt2_batch(train_data, rng)
      loss, grads = value_grad.call(input_batch, target_batch)
      optimizer.update(model, grads)
      MLX::Core.eval(loss)
      if (idx + 1) == @warmup || ((idx + 1) % warmup_every).zero?
        puts "[ruby/karpathy_gpt2] warmup #{idx + 1}/#{@warmup}"
      end
    end

    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    @iterations.times do |idx|
      input_batch, target_batch = next_karpathy_gpt2_batch(train_data, rng)
      loss, grads = value_grad.call(input_batch, target_batch)
      optimizer.update(model, grads)
      MLX::Core.eval(loss)
      if (idx + 1) == @iterations || ((idx + 1) % iter_every).zero?
        puts "[ruby/karpathy_gpt2] iter #{idx + 1}/#{@iterations}"
      end
    end
    finish = Process.clock_gettime(Process::CLOCK_MONOTONIC)

    {
      "average_ms" => (finish - start) * 1000.0 / @iterations,
      "iterations" => @iterations,
      "warmup" => @warmup,
      "output_shape" => output_shape
    }
  end

  def prepare_karpathy_gpt2_dataset
    data_path = karpathy_gpt2_dataset_path
    unless File.exist?(data_path)
      raise "Karpathy GPT-2 fixture missing at #{data_path}. " \
            "Add test/fixtures/karpathy_gpt2_input.txt before running this benchmark."
    end

    text = File.read(data_path)
    bytes = text.bytes
    raise "Karpathy GPT-2 dataset at #{data_path} is empty." if bytes.empty?

    vocab = bytes.uniq.sort
    encode = {}
    vocab.each_with_index { |value, index| encode[value] = index }
    encoded = bytes.map { |value| encode.fetch(value) }
    split = (encoded.length * 9) / 10

    {
      "train" => encoded[0...split],
      "val" => encoded[split...encoded.length],
      "vocab_size" => vocab.length
    }
  end

  def next_karpathy_gpt2_batch(train_data, rng)
    max_start = train_data.length - @sequence_length - 1
    raise "Tiny Shakespeare dataset is too short for block size #{@sequence_length}." if max_start <= 0

    starts = Array.new(@batch_size) { rng.rand(max_start) }
    inputs = starts.map { |start| train_data[start, @sequence_length] }
    targets = starts.map { |start| train_data[(start + 1), @sequence_length] }
    [MLX::Core.array(inputs, MLX::Core.int32), MLX::Core.array(targets, MLX::Core.int32)]
  end

  def karpathy_gpt2_dataset_path
    File.join(@repo_root, "test", "fixtures", "karpathy_gpt2_input.txt")
  end

  def benchmark_ruby_transformer
    src = MLX::Core.random_uniform(
      [@batch_size, @sequence_length, @dims],
      -1.0,
      1.0,
      DEFAULT_DTYPE
    )
    tgt = MLX::Core.random_uniform(
      [@batch_size, @target_sequence_length, @dims],
      -1.0,
      1.0,
      DEFAULT_DTYPE
    )
    src_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(@sequence_length)
    tgt_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(@target_sequence_length)

    model = MLX::NN::Transformer.new(
      dims: @dims,
      num_heads: @num_heads,
      num_encoder_layers: @num_layers,
      num_decoder_layers: @num_layers,
      mlp_dims: @dims * 4,
      dropout: 0.0
    )

    benchmark_ruby_loop("transformer") do
      model.call(src, tgt, src_mask, tgt_mask, nil)
    end
  end

  def benchmark_ruby_cnn
    input = MLX::Core.random_uniform(
      [@batch_size, CNN_HEIGHT, CNN_WIDTH, CNN_CHANNELS],
      -1.0,
      1.0,
      DEFAULT_DTYPE
    )
    flattened_features = 32 * (CNN_HEIGHT / 4) * (CNN_WIDTH / 4)
    conv1 = MLX::NN::Conv2d.new(CNN_CHANNELS, 16, 3, stride: 1, padding: 1, bias: true)
    conv2 = MLX::NN::Conv2d.new(16, 32, 3, stride: 1, padding: 1, bias: true)
    relu = MLX::NN::ReLU.new
    pool = MLX::NN::MaxPool2d.new(2, stride: 2, padding: 0)
    linear = MLX::NN::Linear.new(flattened_features, CNN_CLASSES)

    benchmark_ruby_loop("cnn") do
      y = conv1.call(input)
      y = relu.call(y)
      y = pool.call(y)
      y = conv2.call(y)
      y = relu.call(y)
      y = pool.call(y)
      y = MLX::Core.reshape(y, [@batch_size, flattened_features])
      linear.call(y)
    end
  end

  def benchmark_ruby_mlp
    input = MLX::Core.random_uniform(
      [@batch_size, mlp_input_size],
      -1.0,
      1.0,
      DEFAULT_DTYPE
    )
    relu = MLX::NN::ReLU.new
    layer1 = MLX::NN::Linear.new(mlp_input_size, mlp_hidden_size)
    layer2 = MLX::NN::Linear.new(mlp_hidden_size, mlp_hidden_size)
    layer3 = MLX::NN::Linear.new(mlp_hidden_size, mlp_output_size)

    benchmark_ruby_loop("mlp") do
      y = layer1.call(input)
      y = relu.call(y)
      y = layer2.call(y)
      y = relu.call(y)
      layer3.call(y)
    end
  end

  def benchmark_ruby_rnn
    input = MLX::Core.random_uniform(
      [@batch_size, @sequence_length, @dims],
      -1.0,
      1.0,
      DEFAULT_DTYPE
    )
    rnn = MLX::NN::RNN.new(@dims, rnn_hidden_size)

    benchmark_ruby_loop("rnn") do
      rnn.call(input)
    end
  end

  def benchmark_ruby_loop(label)
    start = nil
    finish = nil
    output = nil
    warmup_every = log_interval(@warmup)
    iter_every = log_interval(@iterations)

    @warmup.times do |idx|
      output = yield
      MLX::Core.eval(output)
      if (idx + 1) == @warmup || ((idx + 1) % warmup_every).zero?
        puts "[ruby/#{label}] warmup #{idx + 1}/#{@warmup}"
      end
    end

    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    @iterations.times do |idx|
      output = yield
      MLX::Core.eval(output)
      if (idx + 1) == @iterations || ((idx + 1) % iter_every).zero?
        puts "[ruby/#{label}] iter #{idx + 1}/#{@iterations}"
      end
    end
    finish = Process.clock_gettime(Process::CLOCK_MONOTONIC)

    {
      "average_ms" => (finish - start) * 1000.0 / @iterations,
      "iterations" => @iterations,
      "warmup" => @warmup,
      "output_shape" => output.shape
    }
  end

  def benchmark_python(model_name)
    python_script = case model_name
    when :transformer
      benchmark_python_transformer_script
    when :cnn
      benchmark_python_cnn_script
    when :mlp
      benchmark_python_mlp_script
    when :rnn
      benchmark_python_rnn_script
    when :karpathy_gpt2
      benchmark_python_karpathy_gpt2_script
    end

    raise "Unknown benchmark model: #{model_name}" if python_script.nil?

    env = {
      "PYTHONPATH" => [File.join(@repo_root, "mlx", "python"), ENV["PYTHONPATH"]].compact.join(File::PATH_SEPARATOR)
    }

    output_lines = []
    status = nil
    Open3.popen2e(env, @python_bin, "-c", python_script, chdir: @repo_root) do |_stdin, stream, wait_thr|
      while (line = stream.gets)
        puts line
        output_lines << line
      end
      status = wait_thr.value
    end

    unless status&.success?
      raise "Python benchmark failed with exit code #{status&.exitstatus}: #{output_lines.join}"
    end

    result_line = output_lines.reverse.find { |line| !line.strip.empty? }
    raise "Python benchmark did not return JSON output: #{output_lines.join}" unless result_line

    JSON.parse(result_line)
  end

  def benchmark_python_transformer_script
    <<~PYTHON
      import json
      import time

      import mlx.core as mx
      from mlx.nn.layers.transformer import Transformer
      from mlx.nn.layers.transformer import MultiHeadAttention

      mx.set_default_device(#{python_device_expr})

      batch_size = #{@batch_size}
      src_seq_len = #{@sequence_length}
      tgt_seq_len = #{@target_sequence_length}
      dims = #{@dims}
      num_heads = #{@num_heads}
      num_layers = #{@num_layers}
      warmup = #{@warmup}
      iterations = #{@iterations}
      warmup_every = max(1, warmup // 5)
      iter_every = max(1, iterations // 5)

      src = mx.random.uniform(
          low=-1.0,
          high=1.0,
          shape=(batch_size, src_seq_len, dims),
          dtype=mx.float32,
      )
      tgt = mx.random.uniform(
          low=-1.0,
          high=1.0,
          shape=(batch_size, tgt_seq_len, dims),
          dtype=mx.float32,
      )
      src_mask = MultiHeadAttention.create_additive_causal_mask(src_seq_len, mx.float32)
      tgt_mask = MultiHeadAttention.create_additive_causal_mask(tgt_seq_len, mx.float32)

      model = Transformer(
          dims=dims,
          num_heads=num_heads,
          num_encoder_layers=num_layers,
          num_decoder_layers=num_layers,
          mlp_dims=dims * 4,
          dropout=0.0,
      )

      for i in range(warmup):
          out = model(src, tgt, src_mask, tgt_mask, None)
          mx.eval(out)
          if (i + 1) == warmup or (i + 1) % warmup_every == 0:
              print(f"[python/transformer] warmup {i + 1}/{warmup}", flush=True)

      start = time.perf_counter()
      for i in range(iterations):
          out = model(src, tgt, src_mask, tgt_mask, None)
          mx.eval(out)
          if (i + 1) == iterations or (i + 1) % iter_every == 0:
              print(f"[python/transformer] iter {i + 1}/{iterations}", flush=True)
      elapsed = time.perf_counter() - start

      print(
          json.dumps(
              {
                  "average_ms": (elapsed / iterations) * 1000.0,
                  "iterations": iterations,
                  "warmup": warmup,
                  "output_shape": list(out.shape),
              }
          )
      )
    PYTHON
  end

  def benchmark_python_karpathy_gpt2_script
    <<~PYTHON
      import json
      import os
      import random
      import time

      import mlx.core as mx
      from mlx.nn import losses
      from mlx.nn.layers.transformer import MultiHeadAttention, TransformerEncoderLayer
      from mlx.nn.layers.embedding import Embedding
      from mlx.nn.layers.linear import Linear
      from mlx.nn.layers.normalization import LayerNorm
      from mlx.nn.layers.dropout import Dropout
      from mlx.nn.utils import value_and_grad
      import mlx.optimizers as optim
      from mlx import nn

      mx.set_default_device(#{python_device_expr})

      data_path = #{karpathy_gpt2_dataset_path.inspect}
      batch_size = #{@batch_size}
      block_size = #{@sequence_length}
      dims = #{@dims}
      num_heads = #{@num_heads}
      num_layers = #{@num_layers}
      warmup = #{@warmup}
      iterations = #{@iterations}
      warmup_every = max(1, warmup // 5)
      iter_every = max(1, iterations // 5)

      if not os.path.exists(data_path):
          raise FileNotFoundError(
              f"Karpathy GPT-2 fixture missing at {data_path}. "
              "Add test/fixtures/karpathy_gpt2_input.txt before running this benchmark."
          )

      with open(data_path, "r", encoding="utf-8") as f:
          text = f.read()

      chars = sorted(set(text))
      if len(chars) == 0:
          raise ValueError(f"Dataset at {data_path} is empty.")

      stoi = {char: idx for idx, char in enumerate(chars)}
      data = [stoi[char] for char in text]
      vocab_size = len(chars)
      split = len(data) * 9 // 10
      train_data = data[:split]

      class GPT2Model(nn.Module):
        def __init__(self):
          super().__init__()
          self.tok_embedding = Embedding(vocab_size, dims)
          self.pos_embedding = Embedding(block_size, dims)
          self.dropout = Dropout(0.0)
          self.blocks = [
            TransformerEncoderLayer(
              dims,
              num_heads,
              mlp_dims=dims * 4,
              dropout=0.0,
              norm_first=True,
            )
            for _ in range(num_layers)
          ]
          self.ln = LayerNorm(dims)
          self.lm_head = Linear(dims, vocab_size)
          self.causal_mask = MultiHeadAttention.create_additive_causal_mask(block_size, mx.float32)

        def __call__(self, idx):
          seq_len = idx.shape[1]
          positions = mx.arange(0, seq_len, dtype=mx.int32)
          x = self.tok_embedding(idx) + self.pos_embedding(positions)
          x = self.dropout(x)
          for block in self.blocks:
            x = block(x, self.causal_mask)
          x = self.ln(x)
          return self.lm_head(x)

      def get_batch():
        max_start = len(train_data) - block_size - 1
        if max_start <= 0:
            raise ValueError(f"Sequence length {block_size} is too large for dataset size {len(train_data)}.")
        starts = [random.randrange(max_start) for _ in range(batch_size)]
        x = [train_data[start : start + block_size] for start in starts]
        y = [train_data[start + 1 : start + block_size + 1] for start in starts]
        return mx.array(x, mx.int32), mx.array(y, mx.int32)

      def loss_fn(x, y):
        logits = model(x)
        batch_size, seq_len, vocab_size = logits.shape
        logits = mx.reshape(logits[:, : (seq_len - 1), :], (batch_size * (seq_len - 1), vocab_size))
        targets = mx.reshape(y[:, 1:seq_len], (batch_size * (seq_len - 1),))
        return losses.cross_entropy(logits, targets, reduction="mean")

      model = GPT2Model()
      step = value_and_grad(model, loss_fn)
      optimizer = optim.AdamW(learning_rate=1e-3)

      sample_input, _ = get_batch()
      out = model(sample_input)

      for i in range(warmup):
        x, y = get_batch()
        loss, grads = step(x, y)
        optimizer.update(model, grads)
        mx.eval(loss)
        if (i + 1) == warmup or (i + 1) % warmup_every == 0:
          print(f"[python/karpathy_gpt2] warmup {i + 1}/{warmup}", flush=True)

      start = time.perf_counter()
      for i in range(iterations):
        x, y = get_batch()
        loss, grads = step(x, y)
        optimizer.update(model, grads)
        mx.eval(loss)
        if (i + 1) == iterations or (i + 1) % iter_every == 0:
          print(f"[python/karpathy_gpt2] iter {i + 1}/{iterations}", flush=True)
      elapsed = time.perf_counter() - start

      print(
          json.dumps(
              {
                  "average_ms": (elapsed / iterations) * 1000.0,
                  "iterations": iterations,
                  "warmup": warmup,
                  "output_shape": list(out.shape),
              }
          )
      )
    PYTHON
  end

  def benchmark_python_cnn_script
    flattened = 32 * (CNN_HEIGHT / 4) * (CNN_WIDTH / 4)
    <<~PYTHON
      import json
      import time

      import mlx.core as mx
      from mlx.nn.layers.convolution import Conv2d
      from mlx.nn.layers.pooling import MaxPool2d
      from mlx.nn.layers.activations import ReLU
      from mlx.nn.layers.linear import Linear

      mx.set_default_device(#{python_device_expr})

      batch_size = #{@batch_size}
      channels = #{CNN_CHANNELS}
      height = #{CNN_HEIGHT}
      width = #{CNN_WIDTH}
      classes = #{CNN_CLASSES}
      warmup = #{@warmup}
      iterations = #{@iterations}
      warmup_every = max(1, warmup // 5)
      iter_every = max(1, iterations // 5)

      conv1 = Conv2d(channels, 16, 3, stride=1, padding=1)
      conv2 = Conv2d(16, 32, 3, stride=1, padding=1)
      relu = ReLU()
      pool = MaxPool2d(2, stride=2)
      linear = Linear(#{flattened}, classes)

      x = mx.random.uniform(
          low=-1.0,
          high=1.0,
          shape=(batch_size, height, width, channels),
          dtype=mx.float32,
      )

      for i in range(warmup):
          y = conv1(x)
          y = relu(y)
          y = pool(y)
          y = conv2(y)
          y = relu(y)
          y = pool(y)
          y = mx.reshape(y, (batch_size, #{flattened}))
          out = linear(y)
          mx.eval(out)
          if (i + 1) == warmup or (i + 1) % warmup_every == 0:
              print(f"[python/cnn] warmup {i + 1}/{warmup}", flush=True)

      start = time.perf_counter()
      for i in range(iterations):
          y = conv1(x)
          y = relu(y)
          y = pool(y)
          y = conv2(y)
          y = relu(y)
          y = pool(y)
          y = mx.reshape(y, (batch_size, #{flattened}))
          out = linear(y)
          mx.eval(out)
          if (i + 1) == iterations or (i + 1) % iter_every == 0:
              print(f"[python/cnn] iter {i + 1}/{iterations}", flush=True)
      elapsed = time.perf_counter() - start

      print(
          json.dumps(
              {
                  "average_ms": (elapsed / iterations) * 1000.0,
                  "iterations": iterations,
                  "warmup": warmup,
                  "output_shape": list(out.shape),
              }
          )
      )
    PYTHON
  end

  def benchmark_python_mlp_script
    <<~PYTHON
      import json
      import time

      import mlx.core as mx
      from mlx.nn.layers.linear import Linear
      from mlx.nn.layers.activations import ReLU

      mx.set_default_device(#{python_device_expr})

      batch_size = #{@batch_size}
      input_size = #{mlp_input_size}
      hidden_size = #{mlp_hidden_size}
      output_size = #{mlp_output_size}
      warmup = #{@warmup}
      iterations = #{@iterations}
      warmup_every = max(1, warmup // 5)
      iter_every = max(1, iterations // 5)

      layer1 = Linear(input_size, hidden_size)
      layer2 = Linear(hidden_size, hidden_size)
      layer3 = Linear(hidden_size, output_size)
      relu = ReLU()

      x = mx.random.uniform(
          low=-1.0,
          high=1.0,
          shape=(batch_size, input_size),
          dtype=mx.float32,
      )

      for i in range(warmup):
          y = layer1(x)
          y = relu(y)
          y = layer2(y)
          y = relu(y)
          out = layer3(y)
          mx.eval(out)
          if (i + 1) == warmup or (i + 1) % warmup_every == 0:
              print(f"[python/mlp] warmup {i + 1}/{warmup}", flush=True)

      start = time.perf_counter()
      for i in range(iterations):
          y = layer1(x)
          y = relu(y)
          y = layer2(y)
          y = relu(y)
          out = layer3(y)
          mx.eval(out)
          if (i + 1) == iterations or (i + 1) % iter_every == 0:
              print(f"[python/mlp] iter {i + 1}/{iterations}", flush=True)
      elapsed = time.perf_counter() - start

      print(
          json.dumps(
              {
                  "average_ms": (elapsed / iterations) * 1000.0,
                  "iterations": iterations,
                  "warmup": warmup,
                  "output_shape": list(out.shape),
              }
          )
      )
    PYTHON
  end

  def benchmark_python_rnn_script
    <<~PYTHON
      import json
      import time

      import mlx.core as mx
      from mlx.nn.layers.recurrent import RNN

      mx.set_default_device(#{python_device_expr})

      batch_size = #{@batch_size}
      seq_len = #{@sequence_length}
      input_size = #{@dims}
      hidden_size = #{rnn_hidden_size}
      warmup = #{@warmup}
      iterations = #{@iterations}
      warmup_every = max(1, warmup // 5)
      iter_every = max(1, iterations // 5)

      rnn = RNN(input_size, hidden_size)
      x = mx.random.uniform(
          low=-1.0,
          high=1.0,
          shape=(batch_size, seq_len, input_size),
          dtype=mx.float32,
      )

      for i in range(warmup):
          out = rnn(x)
          mx.eval(out)
          if (i + 1) == warmup or (i + 1) % warmup_every == 0:
              print(f"[python/rnn] warmup {i + 1}/{warmup}", flush=True)

      start = time.perf_counter()
      for i in range(iterations):
          out = rnn(x)
          mx.eval(out)
          if (i + 1) == iterations or (i + 1) % iter_every == 0:
              print(f"[python/rnn] iter {i + 1}/{iterations}", flush=True)
      elapsed = time.perf_counter() - start

      print(
          json.dumps(
              {
                  "average_ms": (elapsed / iterations) * 1000.0,
                  "iterations": iterations,
                  "warmup": warmup,
                  "output_shape": list(out.shape),
              }
          )
      )
    PYTHON
  end

  def with_compute_device
    default_device = MLX::Core.default_device
    target_device = if @compute_device == :cpu
      MLX::Core.cpu
    else
      MLX::Core.gpu
    end
    MLX::Core.set_default_device(target_device)
    yield
  ensure
    MLX::Core.set_default_device(default_device) if default_device
  end

  def log_interval(total)
    [1, total / 5].max
  end
end
