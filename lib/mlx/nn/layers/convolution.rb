# frozen_string_literal: true

module MLX
  module NN
    class Conv1d < Module
      def initialize(
        in_channels,
        out_channels,
        kernel_size,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
        bias: true
      )
        super()

        if (in_channels % groups) != 0
          raise ArgumentError,
                "The number of input channels (#{in_channels}) must be divisible by the number of groups (#{groups})"
        end

        scale = Math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = MLX::Core.uniform([out_channels, kernel_size, in_channels / groups], -scale, scale)
        self.bias = MLX::Core.zeros([out_channels], MLX::Core.float32) if bias

        @stride = stride
        @padding = padding
        @dilation = dilation
        @groups = groups
      end

      def call(x)
        weight_param = @state["weight"]
        bias_param = @state["bias"]
        y = MLX::Core.conv1d(x, weight_param, @stride, @padding, @dilation, @groups)
        bias_param.nil? ? y : (y + bias_param)
      end
    end

    class Conv2d < Module
      def initialize(
        in_channels,
        out_channels,
        kernel_size,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
        bias: true
      )
        super()

        if (in_channels % groups) != 0
          raise ArgumentError,
                "The number of input channels (#{in_channels}) must be divisible by the number of groups (#{groups})"
        end

        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)

        scale = Math.sqrt(1.0 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = MLX::Core.uniform([out_channels, *kernel_size, in_channels / groups], -scale, scale)
        self.bias = MLX::Core.zeros([out_channels], MLX::Core.float32) if bias

        @stride = stride
        @padding = padding
        @dilation = dilation
        @stride_arg = collapse_uniform(@stride)
        @padding_arg = collapse_uniform(@padding)
        @dilation_arg = collapse_uniform(@dilation)
        @groups = groups
      end

      def call(x)
        weight_param = @state["weight"]
        bias_param = @state["bias"]
        y = MLX::Core.conv2d(x, weight_param, @stride_arg, @padding_arg, @dilation_arg, @groups)
        bias_param.nil? ? y : (y + bias_param)
      end

      private

      def pair(value)
        value.is_a?(Integer) ? [value, value] : value
      end

      def collapse_uniform(value)
        return value unless value.is_a?(Array)
        return value if value.empty?

        value.all? { |entry| entry == value[0] } ? value[0] : value
      end
    end

    class Conv3d < Module
      def initialize(
        in_channels,
        out_channels,
        kernel_size,
        stride: 1,
        padding: 0,
        dilation: 1,
        bias: true
      )
        super()

        kernel_size = triple(kernel_size)
        stride = triple(stride)
        padding = triple(padding)

        scale = Math.sqrt(1.0 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        self.weight = MLX::Core.uniform([out_channels, *kernel_size, in_channels], -scale, scale)
        self.bias = MLX::Core.zeros([out_channels], MLX::Core.float32) if bias

        @stride = stride
        @padding = padding
        @dilation = dilation
        @stride_arg = collapse_uniform(@stride)
        @padding_arg = collapse_uniform(@padding)
        @dilation_arg = collapse_uniform(@dilation)
      end

      def call(x)
        weight_param = @state["weight"]
        bias_param = @state["bias"]
        y = MLX::Core.conv3d(x, weight_param, @stride_arg, @padding_arg, @dilation_arg)
        bias_param.nil? ? y : (y + bias_param)
      end

      private

      def triple(value)
        value.is_a?(Integer) ? [value, value, value] : value
      end

      def collapse_uniform(value)
        return value unless value.is_a?(Array)
        return value if value.empty?

        value.all? { |entry| entry == value[0] } ? value[0] : value
      end
    end
  end
end
