# frozen_string_literal: true

module MLX
  module NN
    class Dropout < Module
      def initialize(p = 0.5)
        super()
        unless p >= 0.0 && p < 1.0
          raise ArgumentError, "The dropout probability #{p} is not in [0, 1)"
        end

        @p_keep = 1.0 - p
      end

      def call(x)
        return x if @p_keep == 1.0 || !training

        mask = MLX::Core.bernoulli(@p_keep, x.shape)
        MLX::Core.multiply(MLX::Core.multiply(mask, x), 1.0 / @p_keep)
      end
    end

    class Dropout2d < Module
      def initialize(p = 0.5)
        super()
        unless p >= 0.0 && p < 1.0
          raise ArgumentError, "The dropout probability #{p} is not in [0, 1)"
        end

        @p_keep = 1.0 - p
      end

      def call(x)
        unless [3, 4].include?(x.ndim)
          raise ArgumentError, "Received input with #{x.ndim} dimensions. Expected 3 or 4 dimensions."
        end

        return x if @p_keep == 1.0 || !training

        mask_shape = x.shape.dup
        mask_shape[-2] = 1
        mask_shape[-3] = 1
        mask = MLX::Core.bernoulli(@p_keep, mask_shape)
        MLX::Core.multiply(MLX::Core.multiply(mask, x), 1.0 / @p_keep)
      end
    end

    class Dropout3d < Module
      def initialize(p = 0.5)
        super()
        unless p >= 0.0 && p < 1.0
          raise ArgumentError, "The dropout probability #{p} is not in [0, 1)"
        end

        @p_keep = 1.0 - p
      end

      def call(x)
        unless [4, 5].include?(x.ndim)
          raise ArgumentError, "Received input with #{x.ndim} dimensions. Expected 4 or 5 dimensions."
        end

        return x if @p_keep == 1.0 || !training

        mask_shape = x.shape.dup
        mask_shape[-2] = 1
        mask_shape[-3] = 1
        mask_shape[-4] = 1
        mask = MLX::Core.bernoulli(@p_keep, mask_shape)
        MLX::Core.multiply(MLX::Core.multiply(mask, x), 1.0 / @p_keep)
      end
    end

  end
end
