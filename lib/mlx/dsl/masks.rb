# frozen_string_literal: true

module MLX
  module DSL
    module Masks
      module_function

      def causal(length:, offset: 0, dtype: MLX::Core.float32)
        length = Integer(length)
        offset = Integer(offset)
        raise ArgumentError, "length must be non-negative" if length.negative?
        raise ArgumentError, "offset must be non-negative" if offset.negative?

        rinds = MLX::Core.arange(0, offset + length, 1)
        linds = if offset.zero?
          rinds
        else
          MLX::Core.arange(offset, offset + length, 1)
        end
        lhs = MLX::Core.expand_dims(linds, 1)
        rhs = MLX::Core.expand_dims(rinds, 0)
        mask = MLX::Core.less(lhs, rhs).astype(dtype)
        min_value = if MLX::Core.respond_to?(:finfo)
          MLX::Core.finfo(dtype).min
        else
          -1e9
        end
        MLX::Core.multiply(mask, min_value)
      end
    end
  end
end
