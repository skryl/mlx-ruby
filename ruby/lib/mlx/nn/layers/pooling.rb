# frozen_string_literal: true

module MLX
  module NN
    class PoolBase < Module
      def initialize(pooling_symbol, kernel_size, stride, padding, padding_value)
        super()
        @pooling_symbol = pooling_symbol
        @kernel_size = kernel_size
        @stride = stride
        @padding = padding
        @padding_value = padding_value
      end

      def call(x)
        if @padding.any? { |p| p[0] > 0 }
          x = MLX::Core.pad(x, [[0, 0], *@padding, [0, 0]], nil, @padding_value)
        end
        windows = sliding_windows(x, @kernel_size, @stride)
        reduce_windows(windows)
      end

      private

      def value_or_list(value, n, message)
        if value.is_a?(Array)
          unless value.length == n && value.all? { |v| v.is_a?(Integer) }
            raise ArgumentError, message
          end
          return value
        end

        unless value.is_a?(Integer)
          raise ArgumentError, message
        end
        Array.new(n, value)
      end

      def reduce_windows(windows)
        result = windows
        window_dims = @kernel_size.length
        window_dims.times do
          axis = result.ndim - 2
          result = if @pooling_symbol == :max
            MLX::Core.max(result, axis)
          else
            MLX::Core.mean(result, axis)
          end
        end
        result
      end

      def sliding_windows(x, window_shape, window_strides)
        if x.ndim < 3
          raise ArgumentError,
                "To extract sliding windows at least 1 spatial dimension (3 total) is needed but the input only has #{x.ndim} dimensions."
        end

        spatial_dims = x.shape[1...-1]
        unless spatial_dims.length == window_shape.length && window_shape.length == window_strides.length
          raise ArgumentError,
                "To extract sliding windows the window shapes and strides must have the same number of spatial dimensions as the signal."
        end

        shape = x.shape
        strides = Array.new(shape.length)
        running = 1
        (shape.length - 1).downto(0) do |i|
          strides[i] = running
          running *= shape[i]
        end

        final_shape = [shape[0]]
        spatial_dims.each_with_index do |size, i|
          window = window_shape[i]
          stride = window_strides[i]
          final_shape << ((size - window) / stride + 1)
        end
        final_shape.concat(window_shape)
        final_shape << shape[-1]

        final_strides = [strides[0]]
        spatial_dims.each_with_index do |_size, i|
          final_strides << (strides[i + 1] * window_strides[i])
        end
        final_strides.concat(strides[1...-1])
        final_strides << strides[-1]

        MLX::Core.as_strided(x, final_shape, final_strides)
      end
    end

    class Pool1dBase < PoolBase
      def initialize(pooling_symbol, padding_value, kernel_size, stride = nil, padding = 0)
        class_name = self.class.name.split("::").last
        msg = "[#{class_name}] '%s' must be an integer or a tuple containing 1 integer"
        kernel_size = value_or_list(kernel_size, 1, format(msg, "kernel_size"))
        stride = stride.nil? ? kernel_size : value_or_list(stride, 1, format(msg, "stride"))
        padding = value_or_list(padding, 1, format(msg, "padding")).map { |p| [p, p] }
        super(pooling_symbol, kernel_size, stride, padding, padding_value)
      end
    end

    class Pool2dBase < PoolBase
      def initialize(pooling_symbol, padding_value, kernel_size, stride = nil, padding = 0)
        class_name = self.class.name.split("::").last
        msg = "[#{class_name}] '%s' must be an integer or a tuple containing 2 integers"
        kernel_size = value_or_list(kernel_size, 2, format(msg, "kernel_size"))
        stride = stride.nil? ? kernel_size : value_or_list(stride, 2, format(msg, "stride"))
        padding = value_or_list(padding, 2, format(msg, "padding")).map { |p| [p, p] }
        super(pooling_symbol, kernel_size, stride, padding, padding_value)
      end
    end

    class Pool3dBase < PoolBase
      def initialize(pooling_symbol, padding_value, kernel_size, stride = nil, padding = 0)
        class_name = self.class.name.split("::").last
        msg = "[#{class_name}] '%s' must be an integer or a tuple containing 3 integers"
        kernel_size = value_or_list(kernel_size, 3, format(msg, "kernel_size"))
        stride = stride.nil? ? kernel_size : value_or_list(stride, 3, format(msg, "stride"))
        padding = value_or_list(padding, 3, format(msg, "padding")).map { |p| [p, p] }
        super(pooling_symbol, kernel_size, stride, padding, padding_value)
      end
    end

    class MaxPool1d < Pool1dBase
      def initialize(kernel_size, stride: nil, padding: 0)
        super(:max, -Float::INFINITY, kernel_size, stride, padding)
      end
    end

    class AvgPool1d < Pool1dBase
      def initialize(kernel_size, stride: nil, padding: 0)
        super(:mean, 0.0, kernel_size, stride, padding)
      end
    end

    class MaxPool2d < Pool2dBase
      def initialize(kernel_size, stride: nil, padding: 0)
        super(:max, -Float::INFINITY, kernel_size, stride, padding)
      end
    end

    class AvgPool2d < Pool2dBase
      def initialize(kernel_size, stride: nil, padding: 0)
        super(:mean, 0.0, kernel_size, stride, padding)
      end
    end

    class MaxPool3d < Pool3dBase
      def initialize(kernel_size, stride: nil, padding: 0)
        super(:max, -Float::INFINITY, kernel_size, stride, padding)
      end
    end

    class AvgPool3d < Pool3dBase
      def initialize(kernel_size, stride: nil, padding: 0)
        super(:mean, 0.0, kernel_size, stride, padding)
      end
    end

    remove_const(:Pool1dBase)
    remove_const(:Pool2dBase)
    remove_const(:Pool3dBase)
    remove_const(:PoolBase)
  end
end
