# frozen_string_literal: true

require "mlx"
require_relative "benchmark_digest"

module BenchmarkExamples
  class CnnExample
    CNN_CHANNELS = 3
    CNN_HEIGHT = 64
    CNN_WIDTH = 64
    CNN_CLASSES = 1024

    attr_reader :label

    def initialize(batch_size:, dtype:)
      @label = "cnn"
      @batch_size = batch_size
      @flattened_features = 32 * (CNN_HEIGHT / 4) * (CNN_WIDTH / 4)

      @input = BenchmarkDigest.deterministic_tensor([batch_size, CNN_HEIGHT, CNN_WIDTH, CNN_CHANNELS], dtype, offset: 0)

      @conv1 = MLX::NN::Conv2d.new(CNN_CHANNELS, 16, 3, stride: 1, padding: 1, bias: true)
      @conv2 = MLX::NN::Conv2d.new(16, 32, 3, stride: 1, padding: 1, bias: true)
      @pool = MLX::NN::MaxPool2d.new(2, stride: 2, padding: 0)
      @linear = MLX::NN::Linear.new(@flattened_features, CNN_CLASSES)
      BenchmarkDigest.assign_deterministic_parameters!([@conv1, @conv2, @linear])

      conv1 = @conv1
      conv2 = @conv2
      pool = @pool
      linear = @linear
      input = @input
      batch_size = @batch_size
      flattened_features = @flattened_features
      @run_step = lambda do
        y = conv1.call(input)
        y = MLX::NN.relu(y)
        y = pool.call(y)
        y = conv2.call(y)
        y = MLX::NN.relu(y)
        y = pool.call(y)
        y = MLX::Core.reshape(y, [batch_size, flattened_features])
        linear.call(y)
      end

      @input_shape = @input.shape
      @input_digest = BenchmarkDigest.digest_array(@input)
      @reference_output_digest = BenchmarkDigest.digest_array(run_step)
      @path_signature = "forward_only_eval_output"
    end

    def run_step
      @run_step.call
    end

    def run_step_proc
      @run_step
    end

    def verification_input_digest
      @input_digest
    end

    def verification_input_shape
      @input_shape
    end

    def verification_reference_output_digest
      @reference_output_digest
    end

    def benchmark_path_signature
      @path_signature
    end
  end
end
