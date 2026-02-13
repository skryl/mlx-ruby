# frozen_string_literal: true

module MLX
  module NN
    class << self
      def sum_gradients(group = nil)
        group ||= MLX::Core.init(false, "any")
        if group.size == 1
          ->(x) { x }
        else
          ->(x) { MLX::Core.all_sum(x, group) }
        end
      end

      def shard_inplace(module_obj, sharding, segments: 1, group: nil)
        group ||= MLX::Core.init(false, "any")

        sharding_fn = if sharding.is_a?(String)
          check_sharding(sharding)
          sharding == "all-to-sharded" ? all_to_sharded(segments) : sharded_to_all(segments)
        else
          sharding
        end

        module_obj.update(shard(module_obj.parameters, sharding_fn, group))
        module_obj
      end

      def shard_linear(module_obj, sharding, segments: 1, group: nil)
        check_sharding(sharding)
        fns = {
          ["all-to-sharded", true] => ->(m, s, g) { AllToShardedLinear.from_linear(m, segments: s, group: g) },
          ["all-to-sharded", false] => ->(m, s, g) { QuantizedAllToShardedLinear.from_quantized_linear(m, segments: s, group: g) },
          ["sharded-to-all", true] => ->(m, s, g) { ShardedToAllLinear.from_linear(m, segments: s, group: g) },
          ["sharded-to-all", false] => lambda { |m, s, g|
            QuantizedShardedToAllLinear.from_quantized_linear(m, segments: s, group: g)
          }
        }
        fns.fetch([sharding, module_obj.is_a?(Linear)]).call(module_obj, segments, group)
      end

      private

      def split_segments(weight, segments, axis)
        if segments.is_a?(Integer) || (segments.is_a?(Array) && !segments.empty? && segments[0].is_a?(Integer))
          return MLX::Core.split(weight, segments, axis)
        end

        n = weight.shape[axis]
        indices = segments.map { |s| (s * n).to_i }
        MLX::Core.split(weight, indices, axis)
      end

      def shard(parameters, sharding_predicate, group)
        n = group.size
        r = group.rank
        return parameters if n == 1

        shard_fn = lambda do |path, weight|
          return weight unless weight.is_a?(MLX::Core::Array)

          spec = sharding_predicate.call(path.to_s, weight)
          return weight if spec.nil?

          axis = nil
          segments = 1
          if spec.is_a?(Integer)
            axis = spec
          elsif spec.is_a?(Array) && spec.length == 2
            axis = spec[0]
            segments = spec[1]
          else
            raise ArgumentError, "The sharding function should return int or tuple [axis, segments]"
          end

          part_segments = split_segments(weight, segments, axis)
          shard_parts = part_segments.map do |part|
            split_segments(part, n, axis)[r]
          end
          MLX::Core.concatenate(shard_parts, axis)
        end

        MLX::Utils.tree_map_with_path(shard_fn, parameters)
      end

      def all_to_sharded(segments)
        lambda do |path, weight|
          if path.end_with?("bias")
            [-1, segments]
          else
            [[weight.ndim - 2, 0].max, segments]
          end
        end
      end

      def sharded_to_all(segments)
        lambda do |path, _weight|
          if path.end_with?("bias")
            nil
          else
            [-1, segments]
          end
        end
      end

      def check_sharding(sharding)
        return if %w[all-to-sharded sharded-to-all].include?(sharding)

        raise ArgumentError,
              "Sharding type sharding=#{sharding} not supported, choose one of 'all-to-sharded' or 'sharded-to-all'"
      end
    end

    class AllToShardedLinear < Module
      def initialize(input_dims, output_dims, bias: true, group: nil)
        super()
        @group = group || MLX::Core.init(false, "any")
        n = @group.size
        if (output_dims % n) != 0
          raise ArgumentError, "Cannot shard the output of size #{output_dims} across #{n} devices."
        end

        scale = Math.sqrt(1.0 / input_dims)
        self.weight = MLX::Core.uniform([output_dims / n, input_dims], -scale, scale)
        self.bias = MLX::Core.uniform([output_dims / n], -scale, scale) if bias
      end

      def call(x)
        x = MLX::NN.sum_gradients(@group).call(x)
        out = MLX::Core.matmul(x, weight.T)
        state.key?("bias") ? MLX::Core.add(out, bias) : out
      end

      def self.from_linear(linear_layer, segments: 1, group: nil)
        group ||= MLX::Core.init(false, "any")
        output_dims, input_dims = linear_layer.weight.shape
        sl = new(input_dims, output_dims, bias: linear_layer.state.key?("bias"), group: group)
        sl.update(
          MLX::NN.__send__(
            :shard,
            linear_layer.parameters,
            MLX::NN.__send__(:all_to_sharded, segments),
            group
          )
        )
        sl
      end
    end

    class ShardedToAllLinear < Module
      def initialize(input_dims, output_dims, bias: true, group: nil)
        super()
        @group = group || MLX::Core.init(false, "any")
        n = @group.size
        if (input_dims % n) != 0
          raise ArgumentError, "The input of size #{input_dims} cannot be sharded across #{n} devices."
        end

        scale = Math.sqrt(1.0 / input_dims)
        self.weight = MLX::Core.uniform([output_dims, input_dims / n], -scale, scale)
        self.bias = MLX::Core.uniform([output_dims], -scale, scale) if bias
      end

      def call(x)
        out = MLX::Core.matmul(x, weight.T)
        out = MLX::Core.all_sum(out, @group)
        state.key?("bias") ? MLX::Core.add(out, bias) : out
      end

      def self.from_linear(linear_layer, segments: 1, group: nil)
        group ||= MLX::Core.init(false, "any")
        output_dims, input_dims = linear_layer.weight.shape
        sl = new(input_dims, output_dims, bias: linear_layer.state.key?("bias"), group: group)
        sl.update(
          MLX::NN.__send__(
            :shard,
            linear_layer.parameters,
            MLX::NN.__send__(:sharded_to_all, segments),
            group
          )
        )
        sl
      end
    end

    class QuantizedAllToShardedLinear < Module
      attr_reader :group_size, :bits

      def initialize(input_dims, output_dims, bias: true, group_size: 64, bits: 4, group: nil)
        super()
        @group = group || MLX::Core.init(false, "any")
        @group_size = group_size
        @bits = bits

        n = @group.size
        if (output_dims % n) != 0
          raise ArgumentError, "Cannot shard the output of size #{output_dims} across #{n} devices."
        end

        scale = Math.sqrt(1.0 / input_dims)
        weight = MLX::Core.uniform([output_dims / n, input_dims], -scale, scale)
        q_weight, q_scales, *q_biases = MLX::Core.quantize(weight, group_size, bits)
        self.weight = q_weight
        self.scales = q_scales
        self.biases = q_biases.empty? ? nil : q_biases[0]
        self.bias = MLX::Core.zeros([output_dims / n], MLX::Core.float32) if bias

        freeze
      end

      def unfreeze(*args, **kwargs)
        super(*args, **kwargs)
        freeze(recurse: false)
      end

      def call(x)
        x = MLX::NN.sum_gradients(@group).call(x)
        out = MLX::Core.quantized_matmul(x, weight, scales, self.biases, true, @group_size, @bits)
        state.key?("bias") ? MLX::Core.add(out, bias) : out
      end

      def self.from_quantized_linear(quantized_linear_layer, segments: 1, group: nil)
        group ||= MLX::Core.init(false, "any")
        output_dims, input_packed = quantized_linear_layer.weight.shape
        input_dims = (input_packed * 32) / quantized_linear_layer.bits

        sl = new(
          input_dims,
          output_dims,
          bias: quantized_linear_layer.state.key?("bias"),
          group_size: quantized_linear_layer.group_size,
          bits: quantized_linear_layer.bits,
          group: group
        )
        sl.update(
          MLX::NN.__send__(
            :shard,
            quantized_linear_layer.parameters,
            MLX::NN.__send__(:all_to_sharded, segments),
            group
          )
        )
        sl
      end
    end

    class QuantizedShardedToAllLinear < Module
      attr_reader :group_size, :bits

      def initialize(input_dims, output_dims, bias: true, group_size: 64, bits: 4, group: nil)
        super()
        @group = group || MLX::Core.init(false, "any")
        @group_size = group_size
        @bits = bits

        n = @group.size
        if (input_dims % n) != 0
          raise ArgumentError, "The input of size #{input_dims} cannot be sharded across #{n} devices."
        end

        scale = Math.sqrt(1.0 / input_dims)
        weight = MLX::Core.uniform([output_dims, input_dims / n], -scale, scale)
        q_weight, q_scales, *q_biases = MLX::Core.quantize(weight, group_size, bits)
        self.weight = q_weight
        self.scales = q_scales
        self.biases = q_biases.empty? ? nil : q_biases[0]
        self.bias = MLX::Core.zeros([output_dims], MLX::Core.float32) if bias

        freeze
      end

      def unfreeze(*args, **kwargs)
        super(*args, **kwargs)
        freeze(recurse: false)
      end

      def call(x)
        out = MLX::Core.quantized_matmul(x, weight, scales, self.biases, true, @group_size, @bits)
        out = MLX::Core.all_sum(out, @group)
        state.key?("bias") ? MLX::Core.add(out, bias) : out
      end

      def self.from_quantized_linear(quantized_linear_layer, segments: 1, group: nil)
        group ||= MLX::Core.init(false, "any")
        output_dims, input_packed = quantized_linear_layer.weight.shape
        input_dims = (input_packed * 32) / quantized_linear_layer.bits

        sl = new(
          input_dims,
          output_dims,
          bias: quantized_linear_layer.state.key?("bias"),
          group_size: quantized_linear_layer.group_size,
          bits: quantized_linear_layer.bits,
          group: group
        )
        sl.update(
          MLX::NN.__send__(
            :shard,
            quantized_linear_layer.parameters,
            MLX::NN.__send__(:sharded_to_all, segments),
            group
          )
        )
        sl
      end
    end

  end
end
