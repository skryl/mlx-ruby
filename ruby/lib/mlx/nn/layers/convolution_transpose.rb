# frozen_string_literal: true

module MLX
  module NN
    class ConvTranspose1d < Module
      def initialize(
        in_channels,
        out_channels,
        kernel_size,
        stride: 1,
        padding: 0,
        dilation: 1,
        output_padding: 0,
        bias: true
      )
        super()

        scale = Math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = MLX::Core.uniform([out_channels, kernel_size, in_channels], -scale, scale)
        self.bias = MLX::Core.zeros([out_channels], MLX::Core.float32) if bias

        @stride = stride
        @padding = padding
        @dilation = dilation
        @output_padding = output_padding
      end

      def call(x)
        y = MLX::Core.conv_transpose1d(x, weight, @stride, @padding, @dilation, @output_padding)
        state.key?("bias") ? MLX::Core.add(y, bias) : y
      end
    end

    class ConvTranspose2d < Module
      def initialize(
        in_channels,
        out_channels,
        kernel_size,
        stride: 1,
        padding: 0,
        dilation: 1,
        output_padding: 0,
        bias: true
      )
        super()

        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)
        output_padding = pair(output_padding)

        scale = Math.sqrt(1.0 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = MLX::Core.uniform([out_channels, *kernel_size, in_channels], -scale, scale)
        self.bias = MLX::Core.zeros([out_channels], MLX::Core.float32) if bias

        @stride = stride
        @padding = padding
        @dilation = dilation
        @output_padding = output_padding
      end

      def call(x)
        y = MLX::Core.conv_transpose2d(x, weight, @stride, @padding, @dilation, @output_padding)
        state.key?("bias") ? MLX::Core.add(y, bias) : y
      end

      private

      def pair(value)
        value.is_a?(Integer) ? [value, value] : value
      end
    end

    class ConvTranspose3d < Module
      def initialize(
        in_channels,
        out_channels,
        kernel_size,
        stride: 1,
        padding: 0,
        dilation: 1,
        output_padding: 0,
        bias: true
      )
        super()

        kernel_size = triple(kernel_size)
        stride = triple(stride)
        padding = triple(padding)
        output_padding = triple(output_padding)

        scale = Math.sqrt(1.0 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        self.weight = MLX::Core.uniform([out_channels, *kernel_size, in_channels], -scale, scale)
        self.bias = MLX::Core.zeros([out_channels], MLX::Core.float32) if bias

        @stride = stride
        @padding = padding
        @dilation = dilation
        @output_padding = output_padding
      end

      def call(x)
        y = MLX::Core.conv_transpose3d(x, weight, @stride, @padding, @dilation, @output_padding)
        state.key?("bias") ? MLX::Core.add(y, bias) : y
      end

      private

      def triple(value)
        value.is_a?(Integer) ? [value, value, value] : value
      end
    end
  end
end
