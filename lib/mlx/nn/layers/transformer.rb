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
        @head_dims = dims / num_heads
        @scale = Math.sqrt(1.0 / @head_dims)
        @heads_layout = [0, 2, 1, 3].freeze
        @split_shape = [@num_heads, @head_dims].freeze
        self.query_proj = Linear.new(query_input_dims, dims, bias: bias)
        self.key_proj = Linear.new(key_input_dims, dims, bias: bias)
        self.value_proj = Linear.new(value_input_dims, value_dims, bias: bias)
        self.out_proj = Linear.new(value_dims, value_output_dims, bias: bias)
      end

      def call(queries, keys, values, mask = nil)
        q_was_2d = queries.ndim == 2
        if q_was_2d
          queries, = maybe_batch(queries)
          keys, = maybe_batch(keys)
          values, = maybe_batch(values)
        end

        query_proj = @state["query_proj"]
        key_proj = @state["key_proj"]
        value_proj = @state["value_proj"]
        out_proj = @state["out_proj"]

        queries = query_proj.call(queries)
        keys = key_proj.call(keys)
        values = value_proj.call(values)

        queries = split_heads(queries)
        keys = split_heads(keys)
        values = split_heads(values)

        output = MLX::Core.scaled_dot_product_attention(queries, keys, values, @scale, mask)
        output = MLX::Core.transpose(output, @heads_layout)
        output = MLX::Core.flatten(output, -2, -1)
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
        x = MLX::Core.unflatten(x, -1, @split_shape)
        MLX::Core.transpose(x, @heads_layout)
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

        self.attention = MultiHeadAttention.new(dims, num_heads)
        self.ln1 = LayerNorm.new(dims)
        self.ln2 = LayerNorm.new(dims)
        self.linear1 = Linear.new(dims, mlp_dims)
        self.linear2 = Linear.new(mlp_dims, dims)
        self.dropout1 = Dropout.new(dropout)
        self.dropout2 = Dropout.new(dropout)
        @activation = activation
        @dropout1_identity = dropout == 0.0
        @dropout2_identity = dropout == 0.0
        @norm_first = norm_first
      end

      def call(x, mask)
        attention = @state["attention"]
        ln1 = @state["ln1"]
        ln2 = @state["ln2"]
        linear1 = @state["linear1"]
        linear2 = @state["linear2"]
        dropout1 = @dropout1_identity ? nil : @state["dropout1"]
        dropout2 = @dropout2_identity ? nil : @state["dropout2"]

        if @norm_first
          y = ln1.call(x)
          y = attention.call(y, y, y, mask)
          y = dropout1.call(y) unless dropout1.nil?
          x = x + y

          y = ln2.call(x)
          y = linear1.call(y)
          y = @activation.nil? ? MLX::Core.maximum(y, 0.0) : @activation.call(y)
          y = dropout2.call(y) unless dropout2.nil?
          y = linear2.call(y)
          y = x + y
        else
          y = attention.call(x, x, x, mask)
          y = dropout1.call(y) unless dropout1.nil?
          x = ln1.call(x + y)

          y = linear1.call(x)
          y = @activation.nil? ? MLX::Core.maximum(y, 0.0) : @activation.call(y)
          y = dropout2.call(y) unless dropout2.nil?
          y = linear2.call(y)
          y = ln2.call(x + y)
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
        @layer_fns = if @checkpoint
          layers.map { |layer| MLX::NN.checkpoint(layer, ->(a, b) { layer.call(a, b) }) }
        else
          layers
        end
      end

      def call(x, mask)
        ln = @state["ln"]
        idx = 0
        while idx < @layer_fns.length
          x = @layer_fns[idx].call(x, mask)
          idx += 1
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
        @dropout1_identity = dropout == 0.0
        @dropout2_identity = dropout == 0.0
        @dropout3_identity = dropout == 0.0
        @norm_first = norm_first
      end

      def call(x, memory, x_mask, memory_mask)
        self_attention = @state["self_attention"]
        cross_attention = @state["cross_attention"]
        ln1 = @state["ln1"]
        ln2 = @state["ln2"]
        ln3 = @state["ln3"]
        linear1 = @state["linear1"]
        linear2 = @state["linear2"]
        dropout1 = @dropout1_identity ? nil : @state["dropout1"]
        dropout2 = @dropout2_identity ? nil : @state["dropout2"]
        dropout3 = @dropout3_identity ? nil : @state["dropout3"]

        if @norm_first
          y = ln1.call(x)
          y = self_attention.call(y, y, y, x_mask)
          y = dropout1.call(y) unless dropout1.nil?
          x = x + y

          y = ln2.call(x)
          y = cross_attention.call(y, memory, memory, memory_mask)
          y = dropout2.call(y) unless dropout2.nil?
          x = x + y

          y = ln3.call(x)
          y = linear1.call(y)
          y = @activation.nil? ? MLX::Core.maximum(y, 0.0) : @activation.call(y)
          y = dropout3.call(y) unless dropout3.nil?
          y = linear2.call(y)
          y = x + y
        else
          y = self_attention.call(x, x, x, x_mask)
          y = dropout1.call(y) unless dropout1.nil?
          x = ln1.call(x + y)

          y = cross_attention.call(y, memory, memory, memory_mask)
          y = dropout2.call(y) unless dropout2.nil?
          x = ln2.call(x + y)

          y = linear1.call(x)
          y = @activation.nil? ? MLX::Core.maximum(y, 0.0) : @activation.call(y)
          y = dropout3.call(y) unless dropout3.nil?
          y = linear2.call(y)
          y = ln3.call(x + y)
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
        @layer_fns = if @checkpoint
          layers.map do |layer|
            MLX::NN.checkpoint(layer, ->(a, b, c, d) { layer.call(a, b, c, d) })
          end
        else
          layers
        end
      end

      def call(x, memory, x_mask, memory_mask)
        ln = @state["ln"]
        idx = 0
        while idx < @layer_fns.length
          x = @layer_fns[idx].call(x, memory, x_mask, memory_mask)
          idx += 1
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
        encoder = @state["encoder"]
        decoder = @state["decoder"]
        memory = encoder.call(src, src_mask)
        decoder.call(tgt, memory, tgt_mask, memory_mask)
      end
    end
  end
end
