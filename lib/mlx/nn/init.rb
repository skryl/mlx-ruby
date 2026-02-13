# frozen_string_literal: true

module MLX
  module NN
    module Init
      module_function

      def constant(value = 0.0, dtype: MLX::Core.float32)
        lambda do |array|
          MLX::Core.full(array.shape, value, dtype)
        end
      end

      def normal(mean: 0.0, std: 1.0, dtype: MLX::Core.float32)
        lambda do |array|
          MLX::Core.normal(array.shape, mean, std, dtype)
        end
      end

      def uniform(low: 0.0, high: 1.0, dtype: MLX::Core.float32)
        lambda do |array|
          MLX::Core.uniform(array.shape, low, high, dtype)
        end
      end

      def identity(dtype: MLX::Core.float32)
        lambda do |array|
          if array.ndim != 2 || array.shape[0] != array.shape[1]
            raise ArgumentError, "The input array must be a square matrix but got shape #{array.shape}."
          end
          MLX::Core.eye(array.shape[0], dtype)
        end
      end

      def glorot_normal(dtype: MLX::Core.float32)
        lambda do |array, gain: 1.0|
          fan_in, fan_out = calculate_fan_in_fan_out(array)
          std = gain * Math.sqrt(2.0 / (fan_in + fan_out))
          MLX::Core.normal(array.shape, 0.0, std, dtype)
        end
      end

      def glorot_uniform(dtype: MLX::Core.float32)
        lambda do |array, gain: 1.0|
          fan_in, fan_out = calculate_fan_in_fan_out(array)
          limit = gain * Math.sqrt(6.0 / (fan_in + fan_out))
          MLX::Core.uniform(array.shape, -limit, limit, dtype)
        end
      end

      def he_normal(dtype: MLX::Core.float32)
        lambda do |array, mode: "fan_in", gain: 1.0|
          fan_in, fan_out = calculate_fan_in_fan_out(array)
          fan = fan_for_mode(mode, fan_in, fan_out)
          std = gain / Math.sqrt(fan)
          MLX::Core.normal(array.shape, 0.0, std, dtype)
        end
      end

      def he_uniform(dtype: MLX::Core.float32)
        lambda do |array, mode: "fan_in", gain: 1.0|
          fan_in, fan_out = calculate_fan_in_fan_out(array)
          fan = fan_for_mode(mode, fan_in, fan_out)
          limit = gain * Math.sqrt(3.0 / fan)
          MLX::Core.uniform(array.shape, -limit, limit, dtype)
        end
      end

      def sparse(sparsity: 0.0, mean: 0.0, std: 1.0, dtype: MLX::Core.float32)
        lambda do |array|
          if array.ndim != 2
            raise ArgumentError, "Only tensors with 2 dimensions are supported"
          end

          mask = MLX::Core.less(MLX::Core.random_uniform(array.shape, 0.0, 1.0, dtype), sparsity)
          values = MLX::Core.normal(array.shape, mean, std, dtype)
          MLX::Core.where(mask, MLX::Core.full(array.shape, 0.0, dtype), values)
        end
      end

      def orthogonal(gain: 1.0, dtype: MLX::Core.float32)
        lambda do |array|
          if array.ndim != 2
            raise ArgumentError, "Orthogonal initialization requires a 2D array but got a #{array.ndim}D array."
          end

          rows, cols = array.shape
          if rows >= cols
            rmat = MLX::Core.normal([rows, cols], 0.0, 1.0, MLX::Core.float32)
            q, r = MLX::Core.qr(rmat)
            q = MLX::Core.multiply(q, MLX::Core.sign(MLX::Core.diag(r)))
            MLX::Core.multiply(q, gain).astype(dtype)
          else
            rmat = MLX::Core.normal([cols, rows], 0.0, 1.0, MLX::Core.float32)
            q_t, r_t = MLX::Core.qr(rmat)
            q_t = MLX::Core.multiply(q_t, MLX::Core.sign(MLX::Core.diag(r_t)))
            q = q_t.T
            MLX::Core.multiply(q, gain).astype(dtype)
          end
        end
      end

      def calculate_fan_in_fan_out(array)
        if array.ndim < 2
          raise ArgumentError,
                "Glorot / He initialization requires at least 2 dimensional input but input with #{array.ndim} dimensions."
        end

        fan_in = array.shape[-1]
        fan_out = array.shape[0]
        if array.ndim > 2
          receptive = array.shape[1...-1].reduce(1) { |acc, d| acc * d }
          fan_in *= receptive
          fan_out *= receptive
        end
        [fan_in, fan_out]
      end

      def fan_for_mode(mode, fan_in, fan_out)
        case mode.to_s
        when "fan_in"
          fan_in
        when "fan_out"
          fan_out
        else
          raise ArgumentError, "Invalid mode: #{mode}. Valid modes are: fan_in, fan_out"
        end
      end
    end

    class << self
      %i[
        constant normal uniform identity glorot_normal glorot_uniform
        he_normal he_uniform sparse orthogonal
      ].each do |name|
        define_method(name) { |*args, **kwargs| Init.public_send(name, *args, **kwargs) }
      end
    end
  end
end
