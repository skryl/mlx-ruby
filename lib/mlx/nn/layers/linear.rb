# frozen_string_literal: true

module MLX
  module NN
    class Identity < Module
      def initialize(*args, **kwargs)
        super()
        _ = [args, kwargs]
      end

      def call(x, *args, **kwargs)
        _ = [args, kwargs]
        x
      end
    end

    class Linear < Module
      def initialize(input_dims, output_dims, bias: true)
        super()
        scale = Math.sqrt(1.0 / input_dims)
        self.weight = MLX::Core.uniform([output_dims, input_dims], -scale, scale)
        self.bias = MLX::Core.uniform([output_dims], -scale, scale) if bias
      end

      def call(x)
        out = MLX::Core.matmul(x, weight.T)
        if state.key?("bias")
          MLX::Core.add(out, bias)
        else
          out
        end
      end

      def to_quantized(group_size: nil, bits: nil, mode: "affine", quantize_input: false)
        if quantize_input
          unless %w[nvfp4 mxfp8].include?(mode.to_s)
            raise ArgumentError,
                  "Quantized activations are only supported for 'nvfp4' and 'mxfp8' modes, got #{mode}."
          end

          QQLinear.from_linear(self, group_size, bits, mode: mode)
        else
          QuantizedLinear.from_linear(self, group_size, bits, mode: mode)
        end
      end
    end

    class Bilinear < Module
      def initialize(input1_dims, input2_dims, output_dims, bias: true)
        super()
        scale = Math.sqrt(1.0 / input1_dims)
        self.weight = MLX::Core.uniform([output_dims, input2_dims, input1_dims], -scale, scale)
        self.bias = MLX::Core.uniform([output_dims], -scale, scale) if bias
      end

      def call(x1, x2)
        out_dims, in2_dims, in1_dims = weight.shape

        x_shape = x1.shape[0...-1]
        batch = x1.size / in1_dims
        x1_2d = MLX::Core.reshape(x1, [batch, in1_dims])
        x2_3d = MLX::Core.reshape(x2, [batch, 1, in2_dims])

        w = MLX::Core.reshape(weight, [out_dims * in2_dims, in1_dims])
        y = MLX::Core.matmul(x1_2d, w.T)
        y = MLX::Core.reshape(y, [batch, out_dims, in2_dims])
        y = MLX::Core.swapaxes(y, -2, -1)
        y = MLX::Core.matmul(x2_3d, y)
        y = MLX::Core.squeeze(y, 1)

        out_shape = x_shape.empty? ? [out_dims] : x_shape + [out_dims]
        y = MLX::Core.reshape(y, out_shape)
        y = MLX::Core.add(y, bias) if state.key?("bias")
        y
      end
    end

  end
end
