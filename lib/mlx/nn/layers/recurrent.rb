# frozen_string_literal: true

module MLX
  module NN
    class RNN < Module
      def initialize(input_size, hidden_size, bias: true, nonlinearity: nil)
        super()

        @nonlinearity = nonlinearity || lambda { |z| MLX::NN.tanh(z) }
        unless @nonlinearity.respond_to?(:call)
          raise ArgumentError, "Nonlinearity must be callable. Current value: #{nonlinearity}."
        end

        scale = 1.0 / Math.sqrt(hidden_size)
        @hidden_size = hidden_size
        self.Wxh = MLX::Core.uniform([hidden_size, input_size], -scale, scale)
        self.Whh = MLX::Core.uniform([hidden_size, hidden_size], -scale, scale)
        self.bias = bias ? MLX::Core.uniform([hidden_size], -scale, scale) : nil
      end

      def call(x, hidden = nil)
        x = MLX::Core.matmul(x, self.Wxh.T)
        x = MLX::Core.add(x, self.bias) unless self.bias.nil?

        all_hidden = []
        sequence_axis = x.ndim - 2
        x.shape[sequence_axis].times do |idx|
          step = MLX::Core.take(x, idx, sequence_axis)
          step = MLX::Core.add(step, MLX::Core.matmul(hidden, self.Whh.T)) unless hidden.nil?
          hidden = @nonlinearity.call(step)
          all_hidden << hidden
        end

        MLX::Core.stack(all_hidden, -2)
      end
    end

    class GRU < Module
      def initialize(input_size, hidden_size, bias: true)
        super()

        @hidden_size = hidden_size
        scale = 1.0 / Math.sqrt(hidden_size)
        self.Wx = MLX::Core.uniform([3 * hidden_size, input_size], -scale, scale)
        self.Wh = MLX::Core.uniform([3 * hidden_size, hidden_size], -scale, scale)
        self.b = bias ? MLX::Core.uniform([3 * hidden_size], -scale, scale) : nil
        self.bhn = bias ? MLX::Core.uniform([hidden_size], -scale, scale) : nil
      end

      def call(x, hidden = nil)
        x = MLX::Core.matmul(x, self.Wx.T)
        x = MLX::Core.add(x, self.b) unless self.b.nil?

        x_rz, x_n = MLX::Core.split(x, [2 * @hidden_size], x.ndim - 1)
        all_hidden = []
        sequence_axis = x.ndim - 2

        x.shape[sequence_axis].times do |idx|
          rz = MLX::Core.take(x_rz, idx, sequence_axis)
          h_proj_n = nil

          unless hidden.nil?
            h_proj = MLX::Core.matmul(hidden, self.Wh.T)
            h_proj_rz, h_proj_n = MLX::Core.split(h_proj, [2 * @hidden_size], h_proj.ndim - 1)
            h_proj_n = MLX::Core.add(h_proj_n, self.bhn) unless self.bhn.nil?
            rz = MLX::Core.add(rz, h_proj_rz)
          end

          rz = MLX::Core.sigmoid(rz)
          r, z = MLX::Core.split(rz, 2, rz.ndim - 1)
          n = MLX::Core.take(x_n, idx, sequence_axis)
          n = MLX::Core.add(n, MLX::Core.multiply(r, h_proj_n)) unless hidden.nil?
          n = MLX::Core.tanh(n)

          if hidden.nil?
            hidden = MLX::Core.multiply(MLX::Core.subtract(1.0, z), n)
          else
            hidden = MLX::Core.add(
              MLX::Core.multiply(MLX::Core.subtract(1.0, z), n),
              MLX::Core.multiply(z, hidden)
            )
          end

          all_hidden << hidden
        end

        MLX::Core.stack(all_hidden, -2)
      end
    end

    class LSTM < Module
      def initialize(input_size, hidden_size, bias: true)
        super()

        @hidden_size = hidden_size
        scale = 1.0 / Math.sqrt(hidden_size)
        self.Wx = MLX::Core.uniform([4 * hidden_size, input_size], -scale, scale)
        self.Wh = MLX::Core.uniform([4 * hidden_size, hidden_size], -scale, scale)
        self.bias = bias ? MLX::Core.uniform([4 * hidden_size], -scale, scale) : nil
      end

      def call(x, hidden = nil, cell = nil)
        x = MLX::Core.matmul(x, self.Wx.T)
        x = MLX::Core.add(x, self.bias) unless self.bias.nil?

        all_hidden = []
        all_cell = []
        sequence_axis = x.ndim - 2

        x.shape[sequence_axis].times do |idx|
          ifgo = MLX::Core.take(x, idx, sequence_axis)
          ifgo = MLX::Core.add(ifgo, MLX::Core.matmul(hidden, self.Wh.T)) unless hidden.nil?

          i, f, g, o = MLX::Core.split(ifgo, 4, ifgo.ndim - 1)
          i = MLX::Core.sigmoid(i)
          f = MLX::Core.sigmoid(f)
          g = MLX::Core.tanh(g)
          o = MLX::Core.sigmoid(o)

          cell = if cell.nil?
            MLX::Core.multiply(i, g)
          else
            MLX::Core.add(MLX::Core.multiply(f, cell), MLX::Core.multiply(i, g))
          end
          hidden = MLX::Core.multiply(o, MLX::Core.tanh(cell))

          all_cell << cell
          all_hidden << hidden
        end

        [MLX::Core.stack(all_hidden, -2), MLX::Core.stack(all_cell, -2)]
      end
    end
  end
end
