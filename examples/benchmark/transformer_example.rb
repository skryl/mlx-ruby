# frozen_string_literal: true

require "mlx"
require_relative "benchmark_digest"

module BenchmarkExamples
  class TransformerExample
    attr_reader :label

    def initialize(batch_size:, sequence_length:, target_sequence_length:, dims:, num_heads:, num_layers:, dtype:)
      @label = "transformer"
      @src = BenchmarkDigest.deterministic_tensor([batch_size, sequence_length, dims], dtype, offset: 0)
      @tgt = BenchmarkDigest.deterministic_tensor(
        [batch_size, target_sequence_length, dims],
        dtype,
        offset: BenchmarkDigest::INPUT_BASE_OFFSET
      )
      @src_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(sequence_length, MLX::Core.float32)
      @tgt_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(target_sequence_length, MLX::Core.float32)
      @input_shape = {
        "src" => @src.shape,
        "tgt" => @tgt.shape,
        "src_mask" => @src_mask.shape,
        "tgt_mask" => @tgt_mask.shape
      }

      @input_digest = BenchmarkDigest.digest_value([@src, @tgt, @src_mask, @tgt_mask])

      @model = MLX::NN::Transformer.new(
        dims: dims,
        num_heads: num_heads,
        num_encoder_layers: num_layers,
        num_decoder_layers: num_layers,
        mlp_dims: dims * 4,
        dropout: 0.0
      )
      BenchmarkDigest.assign_deterministic_parameters!(@model)

      model = @model
      src = @src
      tgt = @tgt
      src_mask = @src_mask
      tgt_mask = @tgt_mask
      @run_step = lambda { model.call(src, tgt, src_mask, tgt_mask, nil) }

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
