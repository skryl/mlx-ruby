# frozen_string_literal: true

require "mlx"

module BenchmarkExamples
  class KarpathyGpt2Example
    class KarpathyGpt2Model < MLX::NN::Module
      def initialize(vocab_size:, dims:, num_heads:, num_layers:, block_size:, dropout: 0.0)
        super()

        @token_embedding = MLX::NN::Embedding.new(vocab_size, dims)
        @pos_embedding = MLX::NN::Embedding.new(block_size, dims)
        @dropout = MLX::NN::Dropout.new(dropout)
        @transformer_blocks = Array.new(num_layers) do
          MLX::NN::TransformerEncoderLayer.new(
            dims,
            num_heads,
            mlp_dims: dims * 4,
            dropout: dropout,
            norm_first: true
          )
        end
        @layer_norm = MLX::NN::LayerNorm.new(dims)
        @proj = MLX::NN::Linear.new(dims, vocab_size)
        @causal_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(block_size)
      end

      def call(input_ids)
        positions = MLX::Core.arange(0, input_ids.shape[1], 1, MLX::Core.int32)
        hidden = MLX::Core.add(
          @token_embedding.call(input_ids),
          @pos_embedding.call(positions)
        )
        hidden = @dropout.call(hidden)

        @transformer_blocks.each do |transformer_block|
          hidden = transformer_block.call(hidden, @causal_mask)
        end

        @proj.call(@layer_norm.call(hidden))
      end
    end

    attr_reader :label, :output_shape

    def initialize(batch_size:, sequence_length:, dims:, num_heads:, num_layers:, repo_root:)
      @label = "karpathy_gpt2"
      @batch_size = batch_size
      @sequence_length = sequence_length

      dataset = prepare_dataset(repo_root)
      @train_data = dataset.fetch("train")
      vocab_size = dataset.fetch("vocab_size")

      @model = KarpathyGpt2Model.new(
        vocab_size: vocab_size,
        dims: dims,
        num_heads: num_heads,
        num_layers: num_layers,
        block_size: sequence_length
      )
      @value_grad = MLX::NN.value_and_grad(@model, method(:loss))
      @optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-3)
      @rng = Random.new(0)

      sample_input, = next_batch
      @output_shape = @model.call(sample_input).shape
    end

    def run_step
      input_batch, target_batch = next_batch
      loss_value, grads = @value_grad.call(input_batch, target_batch)
      @optimizer.update(@model, grads)
      loss_value
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
      data_path = File.join(repo_root, "benchmark", "fixtures", "karpathy.txt")
      unless File.exist?(data_path)
        raise "Karpathy GPT-2 fixture missing at #{data_path}. " \
              "Add benchmark/fixtures/karpathy.txt before running this benchmark."
      end

      text = File.read(data_path)
      bytes = text.bytes
      raise "Karpathy GPT-2 dataset at #{data_path} is empty." if bytes.empty?

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

    def next_batch
      max_start = @train_data.length - @sequence_length - 1
      if max_start <= 0
        raise "Tiny Shakespeare dataset is too short for block size #{@sequence_length}."
      end

      starts = Array.new(@batch_size) { @rng.rand(max_start) }
      inputs = starts.map { |start| @train_data[start, @sequence_length] }
      targets = starts.map { |start| @train_data[(start + 1), @sequence_length] }
      [MLX::Core.array(inputs, MLX::Core.int32), MLX::Core.array(targets, MLX::Core.int32)]
    end
  end
end
