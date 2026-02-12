# frozen_string_literal: true

module MLX
  module NN
    class << self
      def sigmoid(x)
        MLX::Core.sigmoid(x)
      end

      def relu(x)
        MLX::Core.maximum(x, 0.0)
      end

      def relu2(x)
        MLX::Core.square(relu(x))
      end

      def relu6(x)
        MLX::Core.minimum(relu(x), 6.0)
      end

      def leaky_relu(x, negative_slope: 0.01)
        MLX::Core.maximum(MLX::Core.multiply(x, negative_slope), x)
      end

      def log_softmax(x, axis: -1)
        lse = MLX::Core.logsumexp(x, axis, true)
        MLX::Core.subtract(x, lse)
      end

      def elu(x, alpha: 1.0)
        negative = MLX::Core.multiply(alpha, MLX::Core.subtract(MLX::Core.exp(x), 1.0))
        MLX::Core.where(MLX::Core.greater(x, 0.0), x, negative)
      end

      def softmax(x, axis: -1)
        MLX::Core.softmax(x, axis)
      end

      def softplus(x)
        MLX::Core.logaddexp(x, 0.0)
      end

      def softsign(x)
        MLX::Core.divide(x, MLX::Core.add(1.0, MLX::Core.abs(x)))
      end

      def softshrink(x, lambd: 0.5)
        mask = MLX::Core.greater(MLX::Core.abs(x), lambd)
        adjusted = MLX::Core.subtract(x, MLX::Core.multiply(MLX::Core.sign(x), lambd))
        MLX::Core.where(mask, adjusted, 0.0)
      end

      def celu(x, alpha: 1.0)
        pos = MLX::Core.maximum(x, 0.0)
        neg = MLX::Core.multiply(
          alpha,
          MLX::Core.subtract(MLX::Core.exp(MLX::Core.divide(MLX::Core.minimum(x, 0.0), alpha)), 1.0)
        )
        MLX::Core.add(pos, neg)
      end

      def silu(x)
        MLX::Core.multiply(x, MLX::Core.sigmoid(x))
      end

      def log_sigmoid(x)
        MLX::Core.multiply(softplus(MLX::Core.multiply(x, -1.0)), -1.0)
      end

      def gelu(x)
        cdf = MLX::Core.divide(
          MLX::Core.add(1.0, MLX::Core.erf(MLX::Core.divide(x, Math.sqrt(2.0)))),
          2.0
        )
        MLX::Core.multiply(x, cdf)
      end

      def gelu_approx(x)
        x3 = MLX::Core.power(x, 3.0)
        inner = MLX::Core.multiply(Math.sqrt(2.0 / Math::PI), MLX::Core.add(x, MLX::Core.multiply(0.044715, x3)))
        MLX::Core.multiply(0.5, MLX::Core.multiply(x, MLX::Core.add(1.0, MLX::Core.tanh(inner))))
      end

      def gelu_fast_approx(x)
        MLX::Core.multiply(x, MLX::Core.sigmoid(MLX::Core.multiply(1.702, x)))
      end

      def glu(x, axis: -1)
        a, b = MLX::Core.split(x, 2, axis)
        MLX::Core.multiply(a, MLX::Core.sigmoid(b))
      end

      def step(x, threshold: 0.0)
        MLX::Core.where(MLX::Core.greater(x, threshold), 1, 0)
      end

      def selu(x)
        MLX::Core.multiply(elu(x, alpha: 1.67326), 1.0507)
      end

      def prelu(x, alpha)
        MLX::Core.add(
          MLX::Core.maximum(0.0, x),
          MLX::Core.multiply(alpha, MLX::Core.minimum(0.0, x))
        )
      end

      def mish(x)
        MLX::Core.multiply(x, MLX::Core.tanh(softplus(x)))
      end

      def hardswish(x)
        max_x_3 = MLX::Core.maximum(MLX::Core.add(x, 3.0), 0.0)
        MLX::Core.multiply(x, MLX::Core.divide(MLX::Core.minimum(max_x_3, 6.0), 6.0))
      end

      def hard_tanh(x, min_val: -1.0, max_val: 1.0)
        MLX::Core.minimum(MLX::Core.maximum(x, min_val), max_val)
      end

      def hard_shrink(x, lambd: 0.5)
        MLX::Core.where(MLX::Core.greater(MLX::Core.abs(x), lambd), x, 0.0)
      end

      def softmin(x, axis: -1)
        softmax(MLX::Core.multiply(x, -1.0), axis: axis)
      end

      def tanh(x)
        MLX::Core.tanh(x)
      end
    end

    class GLU < Module
      def initialize(axis = -1)
        super()
        @axis = axis
      end

      def call(x)
        MLX::NN.glu(x, axis: @axis)
      end
    end

    class Sigmoid < Module
      def call(x)
        MLX::NN.sigmoid(x)
      end
    end

    class Mish < Module
      def call(x)
        MLX::NN.mish(x)
      end
    end

    class ReLU < Module
      def call(x)
        MLX::NN.relu(x)
      end
    end

    class ReLU2 < Module
      def call(x)
        MLX::NN.relu2(x)
      end
    end

    class ReLU6 < Module
      def call(x)
        MLX::NN.relu6(x)
      end
    end

    class LeakyReLU < Module
      def initialize(negative_slope = 1e-2)
        super()
        @negative_slope = negative_slope
      end

      def call(x)
        MLX::NN.leaky_relu(x, negative_slope: @negative_slope)
      end
    end

    class ELU < Module
      def initialize(alpha = 1.0)
        super()
        @alpha = alpha
      end

      def call(x)
        MLX::NN.elu(x, alpha: @alpha)
      end
    end

    class Softmax < Module
      def call(x)
        MLX::NN.softmax(x)
      end
    end

    class Softplus < Module
      def call(x)
        MLX::NN.softplus(x)
      end
    end

    class Softsign < Module
      def call(x)
        MLX::NN.softsign(x)
      end
    end

    class Softshrink < Module
      def initialize(lambd = 0.5)
        super()
        @lambd = lambd
      end

      def call(x)
        MLX::NN.softshrink(x, lambd: @lambd)
      end
    end

    class CELU < Module
      def initialize(alpha = 1.0)
        super()
        @alpha = alpha
      end

      def call(x)
        MLX::NN.celu(x, alpha: @alpha)
      end
    end

    class SiLU < Module
      def call(x)
        MLX::NN.silu(x)
      end
    end

    class LogSoftmax < Module
      def call(x)
        MLX::NN.log_softmax(x)
      end
    end

    class LogSigmoid < Module
      def call(x)
        MLX::NN.log_sigmoid(x)
      end
    end

    class PReLU < Module
      def initialize(num_parameters = 1, init: 0.25)
        super()
        self.weight = MLX::Core.full([num_parameters], init)
      end

      def call(x)
        MLX::NN.prelu(x, weight)
      end
    end

    class GELU < Module
      def initialize(approx = "none")
        super()
        allowed = %w[none precise tanh fast].freeze
        unless allowed.include?(approx)
          raise ArgumentError, "The approximation should be in #{allowed} but '#{approx}' was given"
        end

        @approx = approx
      end

      def call(x)
        if @approx == "none"
          MLX::NN.gelu(x)
        elsif @approx == "precise" || @approx == "tanh"
          MLX::NN.gelu_approx(x)
        else
          MLX::NN.gelu_fast_approx(x)
        end
      end
    end

    class Tanh < Module
      def call(x)
        MLX::NN.tanh(x)
      end
    end

    class Hardswish < Module
      def call(x)
        MLX::NN.hardswish(x)
      end
    end

    class Step < Module
      def initialize(threshold = 0.0)
        super()
        @threshold = threshold
      end

      def call(x)
        MLX::NN.step(x, threshold: @threshold)
      end
    end

    class SELU < Module
      def call(x)
        MLX::NN.selu(x)
      end
    end

    class HardTanh < Module
      def call(x)
        MLX::NN.hard_tanh(x)
      end
    end

    class HardShrink < Module
      def call(x)
        MLX::NN.hard_shrink(x)
      end
    end

    class Softmin < Module
      def call(x)
        MLX::NN.softmin(x)
      end
    end
  end
end
