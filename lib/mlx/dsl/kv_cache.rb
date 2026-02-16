# frozen_string_literal: true

module MLX
  module DSL
    class KVCache
      attr_reader :num_layers

      def initialize(num_layers:)
        @num_layers = Integer(num_layers)
        raise ArgumentError, "num_layers must be non-negative" if @num_layers.negative?

        @layers = Array.new(@num_layers)
      end

      def layer(index)
        @layers.fetch(__dsl_index(index))
      end

      def []=(index, value)
        @layers[__dsl_index(index)] = value
      end

      def offset(layer:)
        state = self.layer(layer)
        return 0 if state.nil?

        keys, = state
        keys.shape[2]
      end

      def append(layer:, keys:, values:)
        idx = __dsl_index(layer)
        current = @layers[idx]
        if current.nil?
          @layers[idx] = [keys, values]
          return @layers[idx]
        end

        key_cache, value_cache = current
        next_keys = MLX::Core.concatenate([key_cache, keys], 2)
        next_values = MLX::Core.concatenate([value_cache, values], 2)
        @layers[idx] = [next_keys, next_values]
      end

      def truncate!(tokens:, layer: nil)
        keep = Integer(tokens)
        if layer.nil?
          @layers.each_index { |idx| __dsl_truncate_layer!(idx, keep) }
        else
          __dsl_truncate_layer!(__dsl_index(layer), keep)
        end
        self
      end

      def reset!(layer: nil)
        if layer.nil?
          @layers.map! { nil }
        else
          @layers[__dsl_index(layer)] = nil
        end
        self
      end

      private

      def __dsl_index(index)
        idx = Integer(index)
        if idx.negative? || idx >= @num_layers
          raise IndexError, "layer index #{idx} out of range (0...#{@num_layers})"
        end

        idx
      end

      def __dsl_truncate_layer!(idx, keep)
        state = @layers[idx]
        return if state.nil?

        if keep <= 0
          @layers[idx] = nil
          return
        end

        keys, values = state
        total = keys.shape[2]
        return if keep >= total

        start = total - keep
        indices = MLX::Core.arange(start, total, 1, MLX::Core.int32)
        trimmed_keys = MLX::Core.take(keys, indices, 2)
        trimmed_values = MLX::Core.take(values, indices, 2)
        @layers[idx] = [trimmed_keys, trimmed_values]
      end
    end
  end
end
