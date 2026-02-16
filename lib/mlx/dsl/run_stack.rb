# frozen_string_literal: true

module MLX
  module DSL
    def self.run_stack(layers, input, cache: nil, **kwargs)
      modules = layers.to_a
      if cache.is_a?(MLX::DSL::KVCache)
        hidden = input
        modules.each_with_index do |layer, index|
          hidden, next_cache = __dsl_run_stack_layer(
            layer,
            hidden,
            kwargs,
            cache: cache.layer(index),
            use_cache: true
          )
          cache[index] = next_cache
        end
        return [hidden, cache]
      end

      use_cache = !cache.nil?
      cache_state = if use_cache
        entries = cache.to_a
        entries.length < modules.length ? entries + Array.new(modules.length - entries.length) : entries.dup
      else
        nil
      end

      hidden = input
      modules.each_with_index do |layer, index|
        layer_cache = use_cache ? cache_state[index] : nil
        hidden, next_cache = __dsl_run_stack_layer(
          layer,
          hidden,
          kwargs,
          cache: layer_cache,
          use_cache: use_cache
        )
        cache_state[index] = next_cache if use_cache
      end

      use_cache ? [hidden, cache_state] : hidden
    end

    def self.__dsl_run_stack_layer(layer, hidden, kwargs, cache:, use_cache:)
      call_kwargs = kwargs.dup
      call_kwargs[:cache] = cache if use_cache
      result = layer.call(hidden, **call_kwargs)

      if use_cache && result.is_a?(Array) && result.length == 2
        [result[0], result[1]]
      else
        [result, cache]
      end
    rescue ArgumentError => e
      if use_cache && e.message.include?("unknown keyword: :cache")
        result = layer.call(hidden, **kwargs)
        if result.is_a?(Array) && result.length == 2
          return [result[0], result[1]]
        end
        return [result, cache]
      end
      raise
    end
    private_class_method :__dsl_run_stack_layer
  end
end
