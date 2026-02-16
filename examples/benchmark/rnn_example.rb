# frozen_string_literal: true

require "mlx"
require_relative "benchmark_digest"

module BenchmarkExamples
  class RnnExample
    RNN_HIDDEN_MULTIPLIER = 2

    attr_reader :label

    def initialize(batch_size:, sequence_length:, dims:, dtype:)
      @label = "rnn"
      hidden_size = dims * RNN_HIDDEN_MULTIPLIER

      @input = BenchmarkDigest.deterministic_tensor([batch_size, sequence_length, dims], dtype, offset: 0)

      @rnn = MLX::NN::RNN.new(dims, hidden_size)
      BenchmarkDigest.assign_deterministic_parameters!(@rnn)

      @input_digest = BenchmarkDigest.digest_array(@input)
      @reference_output_digest = BenchmarkDigest.digest_array(run_step)
      @path_signature = "forward_only_eval_output"
    end

    def run_step
      @rnn.call(@input)
    end

    def verification_input_digest
      @input_digest
    end

    def verification_reference_output_digest
      @reference_output_digest
    end

    def benchmark_path_signature
      @path_signature
    end
  end
end
