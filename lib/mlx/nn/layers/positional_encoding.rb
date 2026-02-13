# frozen_string_literal: true

module MLX
  module NN
    class RoPE < Module
      def initialize(dims, traditional: false, base: 10_000.0, scale: 1.0)
        super()
        @dims = dims
        @traditional = traditional
        @base = base
        @scale = scale
      end

      def call(x, offset: 0)
        MLX::Core.rope(x, @dims, @traditional, @base, @scale, offset)
      end
    end

    class SinusoidalPositionalEncoding < Module
      def initialize(
        dims,
        min_freq: 0.0001,
        max_freq: 1.0,
        scale: nil,
        cos_first: false,
        full_turns: false
      )
        super()

        half_dims = dims / 2
        one_zero = MLX::Core.subtract(
          1.0,
          MLX::Core.divide(
            MLX::Core.arange(0, half_dims, 1, MLX::Core.float32),
            (half_dims - 1).to_f
          )
        )
        min_log = Math.log(min_freq)
        max_log = Math.log(max_freq)

        self._sigmas = MLX::Core.exp(
          MLX::Core.add(MLX::Core.multiply(one_zero, max_log - min_log), min_log)
        )
        self._sigmas = MLX::Core.multiply(_sigmas, 2.0 * Math::PI) if full_turns

        @scale = scale || Math.sqrt(2.0 / dims)
        @cos_first = cos_first
      end

      def call(x)
        y = MLX::Core.multiply(MLX::Core.expand_dims(x, -1), _sigmas)
        cosy = MLX::Core.cos(y)
        siny = MLX::Core.sin(y)
        y = if @cos_first
          MLX::Core.concatenate([cosy, siny], -1)
        else
          MLX::Core.concatenate([siny, cosy], -1)
        end

        if @scale != 1.0
          MLX::Core.multiply(y, @scale)
        else
          y
        end
      end
    end

    class ALiBi < Module
      class << self
        def create_alibi_matrix(
          q_sequence_length:,
          k_sequence_length:,
          num_heads:,
          offset:,
          dtype: MLX::Core.float32
        )
          x1 = MLX::Core.arange(offset, q_sequence_length, 1)
          x2 = MLX::Core.arange(0, k_sequence_length, 1)
          x1_col = MLX::Core.reshape(x1, [x1.shape[0], 1])
          x2_row = MLX::Core.reshape(x2, [1, x2.shape[0]])
          distance = MLX::Core.multiply(MLX::Core.abs(MLX::Core.subtract(x1_col, x2_row)), -1.0)
          distance = MLX::Core.expand_dims(MLX::Core.expand_dims(distance, 0), 1)

          slope = create_alibi_slope(num_heads: num_heads, dtype: dtype)
          MLX::Core.multiply(distance, slope).astype(dtype)
        end

        def create_alibi_slope(num_heads:, dtype:)
          slopes = get_slopes(num_heads)
          out = MLX::Core.array(slopes, dtype)
          MLX::Core.expand_dims(MLX::Core.expand_dims(out, -1), -1)
        end

        private

        def get_slopes(n)
          if integer_log2?(n)
            start = 2.0**(-(2.0**(-(Math.log2(n) - 3.0))))
            Array.new(n) { |i| start * (start**i) }
          else
            closest_power_of_2 = 2**Math.log2(n).floor
            base = get_slopes(closest_power_of_2)
            extras = get_slopes(2 * closest_power_of_2).each_with_index.select { |_, i| i.even? }.map(&:first)
            base + extras.first(n - closest_power_of_2)
          end
        end

        def integer_log2?(n)
          Math.log2(n).to_i == Math.log2(n)
        end
      end

      def call(attention_scores, offset: 0, mask: nil)
        alibi_mask = self.class.create_alibi_matrix(
          q_sequence_length: attention_scores.shape[-2] + offset,
          k_sequence_length: attention_scores.shape[-1],
          num_heads: attention_scores.shape[1],
          offset: offset,
          dtype: attention_scores.dtype
        )
        alibi_mask = MLX::Core.add(alibi_mask, mask) unless mask.nil?
        MLX::Core.add(attention_scores, alibi_mask)
      end
    end
  end
end
