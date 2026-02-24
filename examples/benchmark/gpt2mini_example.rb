# frozen_string_literal: true

require "mlx"
require_relative "benchmark_digest"

module BenchmarkExamples
  class Gpt2MiniExample
    BATCH_STRIDE = 9_973
    BATCH_OFFSET = 7_919

    class Gpt2MiniModel < MLX::NN::Module
      def initialize(vocab_size:, dims:, num_heads:, num_layers:, block_size:, dropout: 0.0)
        super()

        self.tok_embedding = MLX::NN::Embedding.new(vocab_size, dims)
        self.pos_embedding = MLX::NN::Embedding.new(block_size, dims)
        self.dropout = MLX::NN::Dropout.new(dropout)
        self.blocks = Array.new(num_layers) do
          MLX::NN::TransformerEncoderLayer.new(
            dims,
            num_heads,
            mlp_dims: dims * 4,
            dropout: dropout,
            norm_first: true
          )
        end
        self.ln = MLX::NN::LayerNorm.new(dims)
        self.lm_head = MLX::NN::Linear.new(dims, vocab_size)
        self.causal_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(block_size, MLX::Core.float32)
      end

      def call(input_ids)
        positions = MLX::Core.arange(0, input_ids.shape[1], 1, MLX::Core.int32)
        hidden = MLX::Core.add(
          tok_embedding.call(input_ids),
          pos_embedding.call(positions)
        )
        hidden = dropout.call(hidden)

        blocks.each do |transformer_block|
          hidden = transformer_block.call(hidden, causal_mask)
        end

        lm_head.call(ln.call(hidden))
      end
    end

    attr_reader :label, :output_shape

    def initialize(batch_size:, sequence_length:, dims:, num_heads:, num_layers:, repo_root:)
      @label = "gpt2mini"
      @batch_size = batch_size
      @sequence_length = sequence_length
      @step_index = 0

      dataset = prepare_dataset(repo_root)
      @train_data = dataset.fetch("train")
      vocab_size = dataset.fetch("vocab_size")

      @model = Gpt2MiniModel.new(
        vocab_size: vocab_size,
        dims: dims,
        num_heads: num_heads,
        num_layers: num_layers,
        block_size: sequence_length
      )
      BenchmarkDigest.assign_deterministic_parameters!(@model)
      @value_grad = MLX::NN.value_and_grad(@model, method(:loss))
      @optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-3)

      sample_input, sample_target = batch_for_step(0)
      @input_shape = {
        "sample_input" => sample_input.shape,
        "sample_target" => sample_target.shape
      }
      reference_output = @model.call(sample_input)
      @output_shape = reference_output.shape
      @input_digest = BenchmarkDigest.digest_value(
        {
          "train_data" => @train_data,
          "batch_size" => @batch_size,
          "sequence_length" => @sequence_length,
          "batch_stride" => BATCH_STRIDE,
          "batch_offset" => BATCH_OFFSET,
          "sample_input" => sample_input,
          "sample_target" => sample_target
        }
      )
      @reference_output_digest = BenchmarkDigest.digest_array(reference_output)
      @path_signature = "value_and_grad_update_eval_loss"
    end

    def run_step
      input_batch, target_batch = batch_for_step(@step_index)
      @step_index += 1
      loss_value, grads = @value_grad.call(input_batch, target_batch)
      @optimizer.update(@model, grads)
      loss_value
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

    private

    def loss(input_batch, target_batch)
      logits = @model.call(input_batch)
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

    def prepare_dataset(repo_root)
      data_path = File.join(repo_root, "test", "support", "fixtures", "karpathy.txt")
      unless File.exist?(data_path)
        raise "GPT-2 fixture missing at #{data_path}. " \
              "Add test/support/fixtures/karpathy.txt before running this benchmark."
      end

      bytes = File.binread(data_path).bytes
      raise "GPT-2 dataset at #{data_path} is empty." if bytes.empty?

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

    def batch_for_step(step_index)
      max_start = @train_data.length - @sequence_length - 1
      if max_start <= 0
        raise "Tiny Shakespeare dataset is too short for block size #{@sequence_length}."
      end

      starts = deterministic_starts(max_start, step_index)
      inputs = starts.map { |start| @train_data[start, @sequence_length] }
      targets = starts.map { |start| @train_data[(start + 1), @sequence_length] }
      [MLX::Core.array(inputs, MLX::Core.int32), MLX::Core.array(targets, MLX::Core.int32)]
    end

    def deterministic_starts(max_start, step_index)
      base = step_index * BATCH_STRIDE
      Array.new(@batch_size) do |index|
        (base + (index * BATCH_OFFSET)) % max_start
      end
    end
  end
end
