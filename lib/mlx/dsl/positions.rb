# frozen_string_literal: true

module MLX
  module DSL
    module Positions
      module_function

      def ids_like(input_ids, offset: 0, dtype: nil)
        shape = input_ids.shape
        seq_len = shape[-1]
        dtype ||= input_ids.respond_to?(:dtype) ? input_ids.dtype : MLX::Core.int32

        base = MLX::Core.arange(offset.to_i, offset.to_i + seq_len, 1, dtype)
        return base if shape.length == 1

        reshape_dims = Array.new(shape.length, 1)
        reshape_dims[-1] = seq_len
        expanded = MLX::Core.reshape(base, reshape_dims)
        MLX::Core.broadcast_to(expanded, shape)
      end

      def offset_from_cache(cache, layer: 0)
        return 0 if cache.nil?
        return cache.offset(layer: layer) if cache.respond_to?(:offset)

        if cache.respond_to?(:[]) && !cache[layer].nil?
          keys, = cache[layer]
          return keys.shape[2]
        end

        0
      end
    end
  end
end
