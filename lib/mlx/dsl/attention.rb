# frozen_string_literal: true

module MLX
  module DSL
    class Attention < MLX::NN::Module
      def initialize(
        dims:,
        num_heads:,
        kv_heads: nil,
        qkv_bias: false,
        backend: :sdpa,
        rope: nil,
        cache: false
      )
        super()

        @dims = Integer(dims)
        @num_heads = Integer(num_heads)
        @kv_heads = kv_heads.nil? ? @num_heads : Integer(kv_heads)
        if (@dims % @num_heads) != 0
          raise ArgumentError, "dims must be divisible by num_heads"
        end
        if (@num_heads % @kv_heads) != 0
          raise ArgumentError, "num_heads must be divisible by kv_heads"
        end

        @head_dim = @dims / @num_heads
        @kv_repeats = @num_heads / @kv_heads
        @backend = backend.to_sym
        @cache_enabled = !!cache
        @scale = Math.sqrt(1.0 / @head_dim)

        self.query_proj = MLX::NN::Linear.new(@dims, @num_heads * @head_dim, bias: qkv_bias)
        self.key_proj = MLX::NN::Linear.new(@dims, @kv_heads * @head_dim, bias: qkv_bias)
        self.value_proj = MLX::NN::Linear.new(@dims, @kv_heads * @head_dim, bias: qkv_bias)
        self.out_proj = MLX::NN::Linear.new(@num_heads * @head_dim, @dims, bias: qkv_bias)
        self.rope = __dsl_build_rope(rope)
      end

      def call(queries, keys = nil, values = nil, mask: nil, cache: nil)
        keys ||= queries
        values ||= keys
        q_was_2d = queries.ndim == 2

        queries = MLX::Core.expand_dims(queries, 0) if q_was_2d
        keys = MLX::Core.expand_dims(keys, 0) if keys.ndim == 2
        values = MLX::Core.expand_dims(values, 0) if values.ndim == 2

        batch_size, q_len, = queries.shape

        q = __dsl_pack_heads(query_proj.call(queries), @num_heads)
        k = __dsl_pack_heads(key_proj.call(keys), @kv_heads)
        v = __dsl_pack_heads(value_proj.call(values), @kv_heads)

        offset = cache.nil? ? 0 : cache[0].shape[2]
        if !rope.nil?
          if offset.zero?
            q = rope.call(q)
            k = rope.call(k)
          else
            q = rope.call(q, offset: offset)
            k = rope.call(k, offset: offset)
          end
        end

        unless cache.nil?
          key_cache, value_cache = cache
          k = MLX::Core.concatenate([key_cache, k], 2)
          v = MLX::Core.concatenate([value_cache, v], 2)
        end
        next_cache = [k, v]

        k_for_attn = __dsl_repeat_kv(k)
        v_for_attn = __dsl_repeat_kv(v)
        out = __dsl_attention(q, k_for_attn, v_for_attn, mask)
        out = MLX::Core.transpose(out, [0, 2, 1, 3])
        out = MLX::Core.reshape(out, [batch_size, q_len, @num_heads * @head_dim])
        out = out_proj.call(out)
        out = MLX::Core.squeeze(out, 0) if q_was_2d

        if @cache_enabled || !cache.nil?
          [out, next_cache]
        else
          out
        end
      end

      private

      def __dsl_build_rope(config)
        return nil if config.nil?

        opts = config.transform_keys(&:to_sym)
        rope_kwargs = {
          traditional: opts.fetch(:traditional, false),
          base: opts.fetch(:base, 10_000.0)
        }
        rope_kwargs[:scale] = opts[:scale] if opts.key?(:scale)
        MLX::NN::RoPE.new(@head_dim, **rope_kwargs)
      end

      def __dsl_pack_heads(x, heads)
        batch, length, = x.shape
        x = MLX::Core.reshape(x, [batch, length, heads, @head_dim])
        MLX::Core.transpose(x, [0, 2, 1, 3])
      end

      def __dsl_repeat_kv(x)
        return x if @kv_repeats == 1

        batch, _heads, length, dim = x.shape
        expanded = MLX::Core.expand_dims(x, 2)
        repeated = MLX::Core.concatenate(Array.new(@kv_repeats, expanded), 2)
        MLX::Core.reshape(repeated, [batch, @num_heads, length, dim])
      end

      def __dsl_attention(q, k, v, mask)
        if @backend == :sdpa && MLX::Core.respond_to?(:scaled_dot_product_attention)
          return MLX::Core.scaled_dot_product_attention(q, k, v, @scale, mask)
        end

        scores = MLX::Core.matmul(
          MLX::Core.multiply(q, @scale),
          MLX::Core.transpose(k, [0, 1, 3, 2])
        )
        scores = MLX::Core.add(scores, mask.astype(scores.dtype)) unless mask.nil?
        probs = MLX::Core.softmax(scores.astype(MLX::Core.float32), -1).astype(scores.dtype)
        MLX::Core.matmul(probs, v)
      end
    end
  end
end
