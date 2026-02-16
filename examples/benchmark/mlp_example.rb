# frozen_string_literal: true

require "mlx"
require_relative "benchmark_digest"

module BenchmarkExamples
  class MlpExample
    MLP_FACTOR = 4

    attr_reader :label

    def initialize(batch_size:, dims:, dtype:)
      @label = "mlp"
      @input_size = dims * MLP_FACTOR
      @hidden_size = dims * MLP_FACTOR
      @output_size = dims

      @input = BenchmarkDigest.deterministic_tensor([batch_size, @input_size], dtype, offset: 0)

      @relu = MLX::NN::ReLU.new
      @layer1 = MLX::NN::Linear.new(@input_size, @hidden_size)
      @layer2 = MLX::NN::Linear.new(@hidden_size, @hidden_size)
      @layer3 = MLX::NN::Linear.new(@hidden_size, @output_size)
      BenchmarkDigest.assign_deterministic_parameters!([@layer1, @layer2, @layer3])

      @input_shape = @input.shape
      @input_digest = BenchmarkDigest.digest_array(@input)
      @reference_output_digest = BenchmarkDigest.digest_array(run_step)
      @path_signature = "forward_only_eval_output"
    end

    def run_step
      y = @layer1.call(@input)
      y = @relu.call(y)
      y = @layer2.call(y)
      y = @relu.call(y)
      @layer3.call(y)
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
