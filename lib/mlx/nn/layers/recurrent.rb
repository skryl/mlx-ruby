# frozen_string_literal: true

module MLX
  module NN
    class RNN < Module
      def initialize(input_size, hidden_size, bias: true, nonlinearity: nil)
        super()

        @nonlinearity = nonlinearity || lambda { |z| MLX::NN.tanh(z) }
        @fast_tanh = nonlinearity.nil?
        @compiled_hidden_step = build_compiled_hidden_step if @fast_tanh
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
        wxh = @state["Wxh"]
        whh = @state["Whh"]
        bias = @state["bias"]
        wxh_t = wxh.T
        whh_t = whh.T

        x = if bias.nil?
          MLX::Core.matmul(x, wxh_t)
        else
          MLX::Core.addmm(bias, x, wxh_t)
        end

        sequence_axis = x.ndim - 2
        sequence_length = x.shape[sequence_axis]
        all_hidden = Array.new(sequence_length)
        idx = 0

        while idx < sequence_length
          step = MLX::Core.take(x, idx, sequence_axis)
          hidden = if hidden.nil?
            if @fast_tanh
              MLX::Core.tanh(step)
            else
              @nonlinearity.call(step)
            end
          elsif @fast_tanh
            compiled_hidden_step(step, hidden, whh_t)
          else
            @nonlinearity.call(MLX::Core.addmm(step, hidden, whh_t))
          end
          all_hidden[idx] = hidden
          idx += 1
        end

        MLX::Core.stack(all_hidden, -2)
      end

      private

      def build_compiled_hidden_step
        MLX::Core.compile(lambda do |step, hidden, whh_t|
          MLX::Core.tanh(MLX::Core.addmm(step, hidden, whh_t))
        end)
      rescue StandardError
        nil
      end

      def compiled_hidden_step(step, hidden, whh_t)
        return MLX::Core.tanh(MLX::Core.addmm(step, hidden, whh_t)) if @compiled_hidden_step.nil?

        @compiled_hidden_step.call(step, hidden, whh_t)
      rescue StandardError
        @compiled_hidden_step = nil
        MLX::Core.tanh(MLX::Core.addmm(step, hidden, whh_t))
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
        wx = @state["Wx"]
        wh = @state["Wh"]
        b = @state["b"]
        bhn = @state["bhn"]
        wx_t = wx.T
        wh_t = wh.T

        x = if b.nil?
          MLX::Core.matmul(x, wx_t)
        else
          MLX::Core.addmm(b, x, wx_t)
        end

        x_rz, x_n = MLX::Core.split(x, [2 * @hidden_size], x.ndim - 1)
        sequence_axis = x.ndim - 2
        sequence_length = x.shape[sequence_axis]
        all_hidden = Array.new(sequence_length)
        idx = 0

        while idx < sequence_length
          rz = MLX::Core.take(x_rz, idx, sequence_axis)
          h_proj_n = nil

          unless hidden.nil?
            h_proj = MLX::Core.matmul(hidden, wh_t)
            h_proj_rz, h_proj_n = MLX::Core.split(h_proj, [2 * @hidden_size], h_proj.ndim - 1)
            h_proj_n = MLX::Core.add(h_proj_n, bhn) unless bhn.nil?
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

          all_hidden[idx] = hidden
          idx += 1
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
        wx = @state["Wx"]
        wh = @state["Wh"]
        bias = @state["bias"]
        wx_t = wx.T
        wh_t = wh.T

        x = if bias.nil?
          MLX::Core.matmul(x, wx_t)
        else
          MLX::Core.addmm(bias, x, wx_t)
        end

        sequence_axis = x.ndim - 2
        sequence_length = x.shape[sequence_axis]
        all_hidden = Array.new(sequence_length)
        all_cell = Array.new(sequence_length)
        idx = 0

        while idx < sequence_length
          ifgo = MLX::Core.take(x, idx, sequence_axis)
          ifgo = MLX::Core.addmm(ifgo, hidden, wh_t) unless hidden.nil?

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

          all_cell[idx] = cell
          all_hidden[idx] = hidden
          idx += 1
        end

        [MLX::Core.stack(all_hidden, -2), MLX::Core.stack(all_cell, -2)]
      end
    end
  end
end
