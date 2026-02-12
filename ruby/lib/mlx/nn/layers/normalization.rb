# frozen_string_literal: true

module MLX
  module NN
    class InstanceNorm < Module
      def initialize(dims, eps: 1e-5, affine: false)
        super()
        if affine
          self.weight = MLX::Core.ones([dims], MLX::Core.float32)
          self.bias = MLX::Core.zeros([dims], MLX::Core.float32)
        end
        @dims = dims
        @eps = eps
      end

      def call(x)
        reduction_axes = (1...(x.ndim - 1)).to_a
        mean = reduce_mean_axes(x, reduction_axes, keepdims: true)
        var = MLX::Core.var(x, reduction_axes, true)
        out = MLX::Core.multiply(
          MLX::Core.subtract(x, mean),
          MLX::Core.rsqrt(MLX::Core.add(var, @eps))
        )
        if state.key?("weight")
          MLX::Core.add(MLX::Core.multiply(weight, out), bias)
        else
          out
        end
      end

      private

      def reduce_mean_axes(x, axes, keepdims: false)
        return x if axes.empty?

        result = x
        axes.sort.each_with_index do |axis, i|
          result = MLX::Core.mean(result, axis - i)
        end
        if keepdims
          axes.sort.each do |axis|
            result = MLX::Core.expand_dims(result, axis)
          end
        end
        result
      end
    end

    class LayerNorm < Module
      def initialize(dims, eps: 1e-5, affine: true, bias: true)
        super()
        if affine
          self.weight = MLX::Core.ones([dims], MLX::Core.float32)
          self.bias = MLX::Core.zeros([dims], MLX::Core.float32) if bias
        end
        @eps = eps
        @dims = dims
      end

      def call(x)
        w = state.key?("weight") ? weight : nil
        b = state.key?("bias") ? bias : nil
        MLX::Core.layer_norm(x, w, b, @eps)
      end
    end

    class RMSNorm < Module
      def initialize(dims, eps: 1e-5)
        super()
        self.weight = MLX::Core.ones([dims], MLX::Core.float32)
        @eps = eps
      end

      def call(x)
        MLX::Core.rms_norm(x, weight, @eps)
      end
    end

    class GroupNorm < Module
      def initialize(num_groups, dims, eps: 1e-5, affine: true, pytorch_compatible: false)
        super()
        if affine
          self.bias = MLX::Core.zeros([dims], MLX::Core.float32)
          self.weight = MLX::Core.ones([dims], MLX::Core.float32)
        end
        @num_groups = num_groups
        @dims = dims
        @eps = eps
        @pytorch_compatible = pytorch_compatible
      end

      def call(x)
        out = if @pytorch_compatible
          pytorch_compatible_group_norm(x)
        else
          group_norm(x)
        end
        if state.key?("weight")
          MLX::Core.add(MLX::Core.multiply(weight, out), bias)
        else
          out
        end
      end

      private

      def pytorch_compatible_group_norm(x)
        batch = x.shape[0]
        rest = x.shape[1...-1]
        dims = x.shape[-1]
        group_size = dims / @num_groups
        feature_count = rest.reduce(1) { |acc, v| acc * v }

        out = MLX::Core.reshape(x, [batch, feature_count, @num_groups, group_size])
        out = MLX::Core.transpose(out, [0, 2, 1, 3])
        out = MLX::Core.reshape(out, [batch, @num_groups, feature_count * group_size])
        out = MLX::Core.layer_norm(out, nil, nil, @eps)
        out = MLX::Core.reshape(out, [batch, @num_groups, feature_count, group_size])
        out = MLX::Core.transpose(out, [0, 2, 1, 3])
        MLX::Core.reshape(out, [batch, *rest, dims])
      end

      def group_norm(x)
        batch = x.shape[0]
        rest = x.shape[1...-1]
        dims = x.shape[-1]
        grouped = x.size / (batch * @num_groups)

        out = MLX::Core.reshape(x, [batch, grouped, @num_groups])
        means = MLX::Core.expand_dims(MLX::Core.mean(out, 1), 1)
        var = MLX::Core.var(out, 1, true)
        out = MLX::Core.multiply(
          MLX::Core.subtract(out, means),
          MLX::Core.rsqrt(MLX::Core.add(var, @eps))
        )
        MLX::Core.reshape(out, [batch, *rest, dims])
      end
    end

    class BatchNorm < Module
      def initialize(num_features, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: true)
        super()

        @num_features = num_features
        @eps = eps
        @momentum = momentum
        @track_running_stats = track_running_stats

        if affine
          self.weight = MLX::Core.ones([num_features], MLX::Core.float32)
          self.bias = MLX::Core.zeros([num_features], MLX::Core.float32)
        end

        if @track_running_stats
          self.running_mean = MLX::Core.zeros([num_features], MLX::Core.float32)
          self.running_var = MLX::Core.ones([num_features], MLX::Core.float32)
          freeze(keys: %w[running_mean running_var], recurse: false)
        end
      end

      def unfreeze(*args, **kwargs)
        super(*args, **kwargs)
        freeze(keys: %w[running_mean running_var], recurse: false) if @track_running_stats
      end

      def call(x)
        if x.ndim < 2 || x.ndim > 4
          raise ArgumentError, "Expected input tensor to have 2, 3 or 4 dimensions, but got #{x.ndim}"
        end

        mean, var = calc_stats(x)
        if training && @track_running_stats
          mu = @momentum
          self.running_mean = MLX::Core.add(
            MLX::Core.multiply(1.0 - mu, running_mean),
            MLX::Core.multiply(mu, mean)
          )
          self.running_var = MLX::Core.add(
            MLX::Core.multiply(1.0 - mu, running_var),
            MLX::Core.multiply(mu, var)
          )
        elsif @track_running_stats
          mean = running_mean
          var = running_var
        end

        out = MLX::Core.multiply(
          MLX::Core.subtract(x, mean),
          MLX::Core.rsqrt(MLX::Core.add(var, @eps))
        )
        if state.key?("weight")
          MLX::Core.add(MLX::Core.multiply(weight, out), bias)
        else
          out
        end
      end

      private

      def calc_stats(x)
        reduction_axes = (0...(x.ndim - 1)).to_a
        mean = reduce_mean_axes(x, reduction_axes)
        var = MLX::Core.var(x, reduction_axes)
        [mean, var]
      end

      def reduce_mean_axes(x, axes)
        result = x
        axes.sort.each_with_index do |axis, i|
          result = MLX::Core.mean(result, axis - i)
        end
        result
      end
    end
  end
end
