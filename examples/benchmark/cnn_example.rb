# frozen_string_literal: true

require "mlx"

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
      @input = MLX::Core.random_uniform([batch_size, CNN_HEIGHT, CNN_WIDTH, CNN_CHANNELS], -1.0, 1.0, dtype)
      @conv1 = MLX::NN::Conv2d.new(CNN_CHANNELS, 16, 3, stride: 1, padding: 1, bias: true)
      @conv2 = MLX::NN::Conv2d.new(16, 32, 3, stride: 1, padding: 1, bias: true)
      @relu = MLX::NN::ReLU.new
      @pool = MLX::NN::MaxPool2d.new(2, stride: 2, padding: 0)
      @linear = MLX::NN::Linear.new(@flattened_features, CNN_CLASSES)
    end

    def run_step
      y = @conv1.call(@input)
      y = @relu.call(y)
      y = @pool.call(y)
      y = @conv2.call(y)
      y = @relu.call(y)
      y = @pool.call(y)
      y = MLX::Core.reshape(y, [@batch_size, @flattened_features])
      @linear.call(y)
    end
  end
end
