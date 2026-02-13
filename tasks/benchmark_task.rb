# frozen_string_literal: true

require "json"
require "open3"
require "fileutils"
LIB_ROOT = File.expand_path("../lib", __dir__)
$LOAD_PATH.unshift(LIB_ROOT) unless $LOAD_PATH.include?(LIB_ROOT)
require "mlx"
EXAMPLES_ROOT = File.expand_path("../examples/benchmark", __dir__)
Dir[File.join(EXAMPLES_ROOT, "*.rb")].sort.each { |path| require path }

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
      example = build_ruby_example(model_name)
      benchmark_ruby_loop(example)
    end
  end

  def build_ruby_example(model_name)
    case model_name
    when :transformer
      BenchmarkExamples::TransformerExample.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        target_sequence_length: @target_sequence_length,
        dims: @dims,
        num_heads: @num_heads,
        num_layers: @num_layers,
        dtype: DEFAULT_DTYPE
      )
    when :cnn
      BenchmarkExamples::CnnExample.new(
        batch_size: @batch_size,
        dtype: DEFAULT_DTYPE
      )
    when :mlp
      BenchmarkExamples::MlpExample.new(
        batch_size: @batch_size,
        dims: @dims,
        dtype: DEFAULT_DTYPE
      )
    when :rnn
      BenchmarkExamples::RnnExample.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        dims: @dims,
        dtype: DEFAULT_DTYPE
      )
    when :karpathy_gpt2
      BenchmarkExamples::KarpathyGpt2Example.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        dims: @dims,
        num_heads: @num_heads,
        num_layers: @num_layers,
        repo_root: @repo_root
      )
    else
      raise "Unknown benchmark model: #{model_name}"
    end
  end

  def benchmark_ruby_loop(example)
    start = nil
    finish = nil
    output = nil
    label = example.label
    warmup_every = log_interval(@warmup)
    iter_every = log_interval(@iterations)

    @warmup.times do |idx|
      output = example.run_step
      MLX::Core.eval(output)
      if (idx + 1) == @warmup || ((idx + 1) % warmup_every).zero?
        puts "[ruby/#{label}] warmup #{idx + 1}/#{@warmup}"
      end
    end

    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    @iterations.times do |idx|
      output = example.run_step
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
      "output_shape" => example.respond_to?(:output_shape) && example.output_shape ? example.output_shape : output.shape
    }
  end

  def karpathy_gpt2_dataset_path
    File.join(@repo_root, "benchmark", "fixtures", "karpathy.txt")
  end

  def benchmark_python(model_name)
    script_path = python_script_path(model_name)
    command = [@python_bin, script_path, *python_script_args(model_name)]

    env = {
      "PYTHONPATH" => [File.join(@repo_root, "mlx", "python"), ENV["PYTHONPATH"]].compact.join(File::PATH_SEPARATOR)
    }

    output_lines = []
    status = nil
    Open3.popen2e(env, *command, chdir: @repo_root) do |_stdin, stream, wait_thr|
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

  def python_script_path(model_name)
    file_name = case model_name
    when :transformer
      "transformer_example.py"
    when :cnn
      "cnn_example.py"
    when :mlp
      "mlp_example.py"
    when :rnn
      "rnn_example.py"
    when :karpathy_gpt2
      "karpathy_gpt2_example.py"
    end

    raise "Unknown benchmark model: #{model_name}" if file_name.nil?

    File.join(@repo_root, "examples", "benchmark", "python", file_name)
  end

  def python_script_args(model_name)
    common = [
      "--device", @compute_device.to_s,
      "--batch-size", @batch_size.to_s,
      "--warmup", @warmup.to_s,
      "--iterations", @iterations.to_s
    ]

    case model_name
    when :transformer
      common + [
        "--source-sequence-length", @sequence_length.to_s,
        "--target-sequence-length", @target_sequence_length.to_s,
        "--dims", @dims.to_s,
        "--num-heads", @num_heads.to_s,
        "--num-layers", @num_layers.to_s
      ]
    when :cnn
      common
    when :mlp
      common + ["--dims", @dims.to_s]
    when :rnn
      common + [
        "--sequence-length", @sequence_length.to_s,
        "--dims", @dims.to_s
      ]
    when :karpathy_gpt2
      common + [
        "--sequence-length", @sequence_length.to_s,
        "--dims", @dims.to_s,
        "--num-heads", @num_heads.to_s,
        "--num-layers", @num_layers.to_s,
        "--dataset-path", karpathy_gpt2_dataset_path
      ]
    else
      raise "Unknown benchmark model: #{model_name}"
    end
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
