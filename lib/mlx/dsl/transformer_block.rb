# frozen_string_literal: true

module MLX
  module DSL
    class FeedForward < MLX::NN::Module
      def initialize(dims:, hidden_dims:, kind: :gelu, bias: false)
        super()
        @kind = kind.to_sym
        @dims = Integer(dims)
        @hidden_dims = Integer(hidden_dims)

        case @kind
        when :swiglu
          self.gate_proj = MLX::NN::Linear.new(@dims, @hidden_dims, bias: bias)
          self.up_proj = MLX::NN::Linear.new(@dims, @hidden_dims, bias: bias)
          self.down_proj = MLX::NN::Linear.new(@hidden_dims, @dims, bias: bias)
        else
          self.in_proj = MLX::NN::Linear.new(@dims, @hidden_dims, bias: bias)
          self.out_proj = MLX::NN::Linear.new(@hidden_dims, @dims, bias: bias)
        end
      end

      def call(x)
        case @kind
        when :swiglu
          gated = MLX::NN.silu(gate_proj.call(x))
          down_proj.call(MLX::Core.multiply(gated, up_proj.call(x)))
        when :relu
          out_proj.call(MLX::NN.relu(in_proj.call(x)))
        else
          out_proj.call(MLX::NN.gelu(in_proj.call(x)))
        end
      end
    end

    class TransformerBlock < MLX::NN::Module
      def initialize(
        dims:,
        num_heads:,
        kv_heads: nil,
        ffn_dims: nil,
        norm: :rms,
        norm_eps: 1e-5,
        ffn: nil,
        rope: nil,
        qkv_bias: false,
        backend: :sdpa,
        cache: false
      )
        super()

        ffn_config = (ffn || {}).transform_keys(&:to_sym)
        ffn_kind = ffn_config.fetch(:kind, :gelu)
        hidden_dims = ffn_dims || ffn_config.fetch(:hidden_dims, Integer(dims) * 4)
        ffn_bias = ffn_config.fetch(:bias, false)

        self.attention_norm = __dsl_build_norm(norm, Integer(dims), norm_eps)
        self.attention = MLX::DSL::Attention.new(
          dims: dims,
          num_heads: num_heads,
          kv_heads: kv_heads,
          qkv_bias: qkv_bias,
          backend: backend,
          rope: rope,
          cache: cache
        )
        self.ffn_norm = __dsl_build_norm(norm, Integer(dims), norm_eps)
        self.feed_forward = MLX::DSL::FeedForward.new(
          dims: dims,
          hidden_dims: hidden_dims,
          kind: ffn_kind,
          bias: ffn_bias
        )

        @cache_enabled = !!cache
      end

      def call(x, mask: nil, cache: nil, **_kwargs)
        attn_input = attention_norm.call(x)
        attn_result = attention.call(attn_input, attn_input, attn_input, mask: mask, cache: cache)
        if attn_result.is_a?(Array) && attn_result.length == 2
          attn_out, next_cache = attn_result
        else
          attn_out = attn_result
          next_cache = cache
        end

        hidden = MLX::Core.add(x, attn_out)
        ffn_out = feed_forward.call(ffn_norm.call(hidden))
        output = MLX::Core.add(hidden, ffn_out)

        if @cache_enabled || !cache.nil?
          [output, next_cache]
        else
          output
        end
      end

      private

      def __dsl_build_norm(kind, dims, eps)
        case kind.to_sym
        when :layer, :layer_norm
          MLX::NN::LayerNorm.new(dims, eps: eps)
        when :rms, :rms_norm
          MLX::NN::RMSNorm.new(dims, eps: eps)
        else
          raise ArgumentError, "unsupported norm kind: #{kind.inspect}"
        end
      end
    end
  end
end
