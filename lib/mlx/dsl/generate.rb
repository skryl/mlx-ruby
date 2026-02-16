# frozen_string_literal: true

module MLX
  module DSL
    class Generate
      def initialize(
        model:,
        tokenizer: nil,
        eos_id: nil,
        sampler: nil,
        mode: :decoder_only,
        decoder_start_id: nil
      )
        @model = model
        @tokenizer = tokenizer
        @eos_id = eos_id
        @sampler = { strategy: :argmax }.merge((sampler || {}).transform_keys(&:to_sym))
        @mode = mode.to_sym
        @decoder_start_id = decoder_start_id
      end

      def each_token(prompt: nil, input_ids: nil, max_tokens: 128, **kwargs)
        return enum_for(__method__, prompt: prompt, input_ids: input_ids, max_tokens: max_tokens, **kwargs) unless block_given?

        case @mode
        when :decoder_only
          __dsl_each_decoder_only(prompt: prompt, input_ids: input_ids, max_tokens: max_tokens, **kwargs) do |id, chunk|
            yield id, chunk
          end
        when :encoder_decoder
          __dsl_each_encoder_decoder(prompt: prompt, input_ids: input_ids, max_tokens: max_tokens, **kwargs) do |id, chunk|
            yield id, chunk
          end
        else
          raise ArgumentError, "unsupported generation mode: #{@mode.inspect}"
        end

        self
      end

      private

      def __dsl_each_decoder_only(prompt:, input_ids:, max_tokens:, **kwargs)
        tokens = input_ids.nil? ? __dsl_encode(prompt) : input_ids
        model_input = __dsl_input_array(tokens)
        logits, cache = __dsl_decode_step(model_input, cache: nil, **kwargs)

        max_tokens.to_i.times do
          token = __dsl_sample(__dsl_last_logits(logits))
          token_id = __dsl_token_id(token)
          chunk = __dsl_decode_token(token_id)
          yield token_id, chunk
          break if !@eos_id.nil? && token_id == @eos_id

          next_input = MLX::Core.array([[token_id]], MLX::Core.int32)
          logits, cache = __dsl_decode_step(next_input, cache: cache, **kwargs)
        end
      end

      def __dsl_each_encoder_decoder(prompt:, input_ids:, max_tokens:, **kwargs)
        tokens = input_ids.nil? ? __dsl_encode(prompt) : input_ids
        source = __dsl_input_array(tokens)

        if @model.respond_to?(:encode) && @model.respond_to?(:decode)
          memory = @model.encode(source)
          start_id = __dsl_decoder_start_id
          decoder_input = MLX::Core.array([[start_id]], MLX::Core.int32)
          cache = nil

          max_tokens.to_i.times do
            decoded = @model.decode(decoder_input, memory, cache: cache, **kwargs)
            logits, cache = __dsl_split_logits_and_cache(decoded, cache)
            token = __dsl_sample(__dsl_last_logits(logits))
            token_id = __dsl_token_id(token)
            chunk = __dsl_decode_token(token_id)
            yield token_id, chunk
            break if !@eos_id.nil? && token_id == @eos_id

            decoder_input = MLX::Core.array([[token_id]], MLX::Core.int32)
          end
          return
        end

        # Fallback path for model.call-style APIs.
        __dsl_each_decoder_only(prompt: prompt, input_ids: tokens, max_tokens: max_tokens, **kwargs) do |id, chunk|
          yield id, chunk
        end
      end

      def __dsl_decode_step(input_ids, cache:, **kwargs)
        output = @model.call(input_ids, cache: cache, **kwargs)
        __dsl_split_logits_and_cache(output, cache)
      end

      def __dsl_split_logits_and_cache(output, fallback_cache)
        if output.is_a?(Array) && output.length == 2
          [output[0], output[1]]
        else
          [output, fallback_cache]
        end
      end

      def __dsl_last_logits(logits)
        return logits if logits.ndim == 2
        return logits if logits.ndim == 1

        index = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
        MLX::Core.squeeze(MLX::Core.take(logits, index, 1), 1)
      end

      def __dsl_sample(logits)
        strategy = @sampler.fetch(:strategy, :argmax).to_sym
        temperature = @sampler.fetch(:temperature, 1.0).to_f

        return MLX::Core.argmax(logits, -1) if strategy == :argmax || temperature.zero?

        case strategy
        when :top_k
          __dsl_top_k_sample(logits, k: Integer(@sampler.fetch(:k, 40)), temperature: temperature)
        when :temperature, :categorical
          __dsl_temperature_sample(logits, temperature: temperature)
        else
          raise ArgumentError, "unsupported sampler strategy: #{strategy.inspect}"
        end
      end

      def __dsl_temperature_sample(logits, temperature:)
        scaled = if temperature == 1.0
          logits
        else
          MLX::Core.multiply(logits, 1.0 / temperature)
        end
        MLX::Core.categorical(scaled)
      end

      def __dsl_top_k_sample(logits, k:, temperature:)
        rows = logits.ndim == 1 ? [logits.to_a] : logits.to_a
        masked = rows.map do |row|
          pairs = row.each_with_index.sort_by { |(value, _index)| -value }
          keep = pairs.first([k, row.length].min).map(&:last)
          filtered = Array.new(row.length, -Float::INFINITY)
          keep.each { |idx| filtered[idx] = row[idx] }
          filtered
        end
        masked_logits = MLX::Core.array(masked, logits.dtype)
        __dsl_temperature_sample(masked_logits, temperature: temperature)
      end

      def __dsl_encode(prompt)
        raise ArgumentError, "prompt/input_ids required when tokenizer is unavailable" if @tokenizer.nil?

        @tokenizer.encode(prompt.to_s)
      end

      def __dsl_input_array(tokens)
        if tokens.is_a?(MLX::Core::Array)
          return tokens if tokens.ndim > 1

          return MLX::Core.expand_dims(tokens.astype(MLX::Core.int32), 0)
        end

        arr = tokens.to_a
        nested = arr.empty? ? [[]] : (arr.first.is_a?(Array) ? arr : [arr])
        MLX::Core.array(nested, MLX::Core.int32)
      end

      def __dsl_token_id(token)
        return token.item.to_i if token.respond_to?(:item)

        value = token.to_a
        if value.is_a?(Array)
          first = value.first
          return first.first.to_i if first.is_a?(Array)
          return first.to_i
        end
        value.to_i
      end

      def __dsl_decode_token(token_id)
        return nil if @tokenizer.nil? || !@tokenizer.respond_to?(:decode)

        @tokenizer.decode([token_id])
      end

      def __dsl_decoder_start_id
        return @decoder_start_id unless @decoder_start_id.nil?
        return @tokenizer.decoder_start_id if !@tokenizer.nil? && @tokenizer.respond_to?(:decoder_start_id)

        raise ArgumentError, "decoder_start_id is required for encoder-decoder mode"
      end
    end
  end
end
