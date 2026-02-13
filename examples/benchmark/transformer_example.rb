# frozen_string_literal: true

require "mlx"

module BenchmarkExamples
  class TransformerExample
    attr_reader :label

    def initialize(batch_size:, sequence_length:, target_sequence_length:, dims:, num_heads:, num_layers:, dtype:)
      @label = "transformer"
      @src = MLX::Core.random_uniform([batch_size, sequence_length, dims], -1.0, 1.0, dtype)
      @tgt = MLX::Core.random_uniform([batch_size, target_sequence_length, dims], -1.0, 1.0, dtype)
      @src_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(sequence_length)
      @tgt_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(target_sequence_length)
      @model = MLX::NN::Transformer.new(
        dims: dims,
        num_heads: num_heads,
        num_encoder_layers: num_layers,
        num_decoder_layers: num_layers,
        mlp_dims: dims * 4,
        dropout: 0.0
      )
    end

    def run_step
      @model.call(@src, @tgt, @src_mask, @tgt_mask, nil)
    end
  end
end
