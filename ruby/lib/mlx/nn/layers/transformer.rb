# frozen_string_literal: true

module MLX
  module NN
    class MultiHeadAttention < Module
      attr_reader :num_heads

      def initialize(
        dims,
        num_heads,
        query_input_dims: nil,
        key_input_dims: nil,
        value_input_dims: nil,
        value_dims: nil,
        value_output_dims: nil,
        bias: false
      )
        super()

        if (dims % num_heads) != 0
          raise ArgumentError,
                "The input feature dimensions should be divisible by the number of heads (#{dims} % #{num_heads}) != 0"
        end

        query_input_dims ||= dims
        key_input_dims ||= dims
        value_input_dims ||= key_input_dims
        value_dims ||= dims
        value_output_dims ||= dims

        @num_heads = num_heads
        self.query_proj = Linear.new(query_input_dims, dims, bias: bias)
        self.key_proj = Linear.new(key_input_dims, dims, bias: bias)
        self.value_proj = Linear.new(value_input_dims, value_dims, bias: bias)
        self.out_proj = Linear.new(value_dims, value_output_dims, bias: bias)
      end

      def call(queries, keys, values, mask = nil)
        queries, q_was_2d = maybe_batch(queries)
        keys, = maybe_batch(keys)
        values, = maybe_batch(values)

        queries = query_proj.call(queries)
        keys = key_proj.call(keys)
        values = value_proj.call(values)

        queries = split_heads(queries)
        keys = split_heads(keys)
        values = split_heads(values)

        scale = Math.sqrt(1.0 / queries.shape[-1])
        output = MLX::Core.scaled_dot_product_attention(queries, keys, values, scale, mask)
        output = MLX::Core.transpose(output, [0, 2, 1, 3])
        output = output.flatten(-2, -1)
        output = out_proj.call(output)
        q_was_2d ? MLX::Core.squeeze(output, 0) : output
      end

      def self.create_additive_causal_mask(n, dtype = MLX::Core.float32)
        indices = MLX::Core.arange(0, n, 1)
        lhs = MLX::Core.reshape(indices, [n, 1])
        rhs = MLX::Core.reshape(indices, [1, n])
        mask = MLX::Core.less(lhs, rhs).astype(dtype)
        MLX::Core.multiply(mask, MLX::Core.finfo(dtype).min)
      end

      private

      def split_heads(x)
        batch, length, dims = x.shape
        head_dim = dims / @num_heads
        x = MLX::Core.reshape(x, [batch, length, @num_heads, head_dim])
        MLX::Core.transpose(x, [0, 2, 1, 3])
      end

      def maybe_batch(x)
        if x.ndim == 2
          [MLX::Core.expand_dims(x, 0), true]
        else
          [x, false]
        end
      end
    end

    class TransformerEncoderLayer < Module
      def initialize(
        dims,
        num_heads,
        mlp_dims: nil,
        dropout: 0.0,
        activation: nil,
        norm_first: true
      )
        super()
        mlp_dims ||= dims * 4
        activation ||= lambda { |x| MLX::NN.relu(x) }

        self.attention = MultiHeadAttention.new(dims, num_heads)
        self.ln1 = LayerNorm.new(dims)
        self.ln2 = LayerNorm.new(dims)
        self.linear1 = Linear.new(dims, mlp_dims)
        self.linear2 = Linear.new(mlp_dims, dims)
        self.dropout1 = Dropout.new(dropout)
        self.dropout2 = Dropout.new(dropout)
        @activation = activation
        @norm_first = norm_first
      end

      def call(x, mask)
        if @norm_first
          y = ln1.call(x)
          y = attention.call(y, y, y, mask)
          y = dropout1.call(y)
          x = MLX::Core.add(x, y)

          y = ln2.call(x)
          y = linear1.call(y)
          y = @activation.call(y)
          y = dropout2.call(y)
          y = linear2.call(y)
          y = MLX::Core.add(x, y)
        else
          y = attention.call(x, x, x, mask)
          y = dropout1.call(y)
          x = ln1.call(MLX::Core.add(x, y))

          y = linear1.call(x)
          y = @activation.call(y)
          y = dropout2.call(y)
          y = linear2.call(y)
          y = ln2.call(MLX::Core.add(x, y))
        end

        y
      end
    end

    class TransformerEncoder < Module
      def initialize(
        num_layers,
        dims,
        num_heads,
        mlp_dims: nil,
        dropout: 0.0,
        activation: nil,
        norm_first: true,
        checkpoint: false
      )
        super()
        activation ||= lambda { |x| MLX::NN.relu(x) }
        self.layers = Array.new(num_layers) do
          TransformerEncoderLayer.new(
            dims,
            num_heads,
            mlp_dims: mlp_dims,
            dropout: dropout,
            activation: activation,
            norm_first: norm_first
          )
        end
        self.ln = LayerNorm.new(dims)
        @checkpoint = checkpoint
      end

      def call(x, mask)
        layers.each do |layer|
          if @checkpoint
            layer_fn = MLX::NN.checkpoint(->(a, b) { layer.call(a, b) })
            x = layer_fn.call(x, mask)
          else
            x = layer.call(x, mask)
          end
        end
        ln.call(x)
      end
    end

    class TransformerDecoderLayer < Module
      def initialize(
        dims,
        num_heads,
        mlp_dims: nil,
        dropout: 0.0,
        activation: nil,
        norm_first: true
      )
        super()
        mlp_dims ||= dims * 4
        activation ||= lambda { |x| MLX::NN.relu(x) }

        self.self_attention = MultiHeadAttention.new(dims, num_heads)
        self.cross_attention = MultiHeadAttention.new(dims, num_heads)
        self.ln1 = LayerNorm.new(dims)
        self.ln2 = LayerNorm.new(dims)
        self.ln3 = LayerNorm.new(dims)
        self.linear1 = Linear.new(dims, mlp_dims)
        self.linear2 = Linear.new(mlp_dims, dims)
        self.dropout1 = Dropout.new(dropout)
        self.dropout2 = Dropout.new(dropout)
        self.dropout3 = Dropout.new(dropout)
        @activation = activation
        @norm_first = norm_first
      end

      def call(x, memory, x_mask, memory_mask)
        if @norm_first
          y = ln1.call(x)
          y = self_attention.call(y, y, y, x_mask)
          y = dropout1.call(y)
          x = MLX::Core.add(x, y)

          y = ln2.call(x)
          y = cross_attention.call(y, memory, memory, memory_mask)
          y = dropout2.call(y)
          x = MLX::Core.add(x, y)

          y = ln3.call(x)
          y = linear1.call(y)
          y = @activation.call(y)
          y = dropout3.call(y)
          y = linear2.call(y)
          y = MLX::Core.add(x, y)
        else
          y = self_attention.call(x, x, x, x_mask)
          y = dropout1.call(y)
          x = ln1.call(MLX::Core.add(x, y))

          y = cross_attention.call(y, memory, memory, memory_mask)
          y = dropout2.call(y)
          x = ln2.call(MLX::Core.add(x, y))

          y = linear1.call(x)
          y = @activation.call(y)
          y = dropout3.call(y)
          y = linear2.call(y)
          y = ln3.call(MLX::Core.add(x, y))
        end

        y
      end
    end

    class TransformerDecoder < Module
      def initialize(
        num_layers,
        dims,
        num_heads,
        mlp_dims: nil,
        dropout: 0.0,
        activation: nil,
        norm_first: true,
        checkpoint: false
      )
        super()
        activation ||= lambda { |x| MLX::NN.relu(x) }
        self.layers = Array.new(num_layers) do
          TransformerDecoderLayer.new(
            dims,
            num_heads,
            mlp_dims: mlp_dims,
            dropout: dropout,
            activation: activation,
            norm_first: norm_first
          )
        end
        self.ln = LayerNorm.new(dims)
        @checkpoint = checkpoint
      end

      def call(x, memory, x_mask, memory_mask)
        layers.each do |layer|
          if @checkpoint
            layer_fn = MLX::NN.checkpoint(->(a, b, c, d) { layer.call(a, b, c, d) })
            x = layer_fn.call(x, memory, x_mask, memory_mask)
          else
            x = layer.call(x, memory, x_mask, memory_mask)
          end
        end
        ln.call(x)
      end
    end

    class Transformer < Module
      def initialize(
        dims: 512,
        num_heads: 8,
        num_encoder_layers: 6,
        num_decoder_layers: 6,
        mlp_dims: nil,
        dropout: 0.0,
        activation: nil,
        custom_encoder: nil,
        custom_decoder: nil,
        norm_first: true,
        checkpoint: false
      )
        super()

        activation ||= lambda { |x| MLX::NN.relu(x) }
        self.encoder = custom_encoder || TransformerEncoder.new(
          num_encoder_layers,
          dims,
          num_heads,
          mlp_dims: mlp_dims,
          dropout: dropout,
          activation: activation,
          norm_first: norm_first,
          checkpoint: checkpoint
        )
        self.decoder = custom_decoder || TransformerDecoder.new(
          num_decoder_layers,
          dims,
          num_heads,
          mlp_dims: mlp_dims,
          dropout: dropout,
          activation: activation,
          norm_first: norm_first,
          checkpoint: checkpoint
        )
      end

      def call(src, tgt, src_mask, tgt_mask, memory_mask)
        memory = encoder.call(src, src_mask)
        decoder.call(tgt, memory, tgt_mask, memory_mask)
      end
    end
  end
end
