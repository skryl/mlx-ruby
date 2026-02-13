# frozen_string_literal: true

require "mlx"

module BenchmarkExamples
  class RnnExample
    RNN_HIDDEN_MULTIPLIER = 2

    attr_reader :label

    def initialize(batch_size:, sequence_length:, dims:, dtype:)
      @label = "rnn"
      hidden_size = dims * RNN_HIDDEN_MULTIPLIER
      @input = MLX::Core.random_uniform([batch_size, sequence_length, dims], -1.0, 1.0, dtype)
      @rnn = MLX::NN::RNN.new(dims, hidden_size)
    end

    def run_step
      @rnn.call(@input)
    end
  end
end
