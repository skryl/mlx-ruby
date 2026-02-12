# frozen_string_literal: true

module MLX
  module NN
    class Embedding < Module
      def initialize(num_embeddings, dims)
        super()
        scale = Math.sqrt(1.0 / dims)
        self.weight = MLX::Core.normal([num_embeddings, dims], 0.0, scale)
      end

      def call(x)
        MLX::Core.take(weight, x, 0)
      end

      def as_linear(x)
        MLX::Core.matmul(x, weight.T)
      end

      def to_quantized(group_size: nil, bits: nil, mode: "affine", quantize_input: false)
        raise ArgumentError, "Quantized input is not supported." if quantize_input

        QuantizedEmbedding.from_embedding(self, group_size, bits, mode: mode)
      end
    end

  end
end
