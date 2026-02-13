# frozen_string_literal: true

require "mlx"

module BenchmarkExamples
  class MlpExample
    MLP_FACTOR = 4

    attr_reader :label

    def initialize(batch_size:, dims:, dtype:)
      @label = "mlp"
      @input_size = dims * MLP_FACTOR
      @hidden_size = dims * MLP_FACTOR
      @output_size = dims

      @input = MLX::Core.random_uniform([batch_size, @input_size], -1.0, 1.0, dtype)
      @relu = MLX::NN::ReLU.new
      @layer1 = MLX::NN::Linear.new(@input_size, @hidden_size)
      @layer2 = MLX::NN::Linear.new(@hidden_size, @hidden_size)
      @layer3 = MLX::NN::Linear.new(@hidden_size, @output_size)
    end

    def run_step
      y = @layer1.call(@input)
      y = @relu.call(y)
      y = @layer2.call(y)
      y = @relu.call(y)
      @layer3.call(y)
    end
  end
end
