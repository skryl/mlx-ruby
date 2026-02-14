# frozen_string_literal: true

module MLX
  module DSL
    module Data
      def self.from(source = nil, &block)
        if !source.nil? && block_given?
          raise ArgumentError, "data pipeline source accepts either a source or block, not both"
        end

        producer = block_given? ? block : source
        if producer.nil?
          raise ArgumentError, "data pipeline requires a source enumerable or source block"
        end

        Pipeline.new(__dsl_factory_for(producer))
      end

      def self.pipeline(source = nil, &block)
        from(source, &block)
      end

      def self.__dsl_factory_for(producer)
        if producer.respond_to?(:call)
          lambda do
            __dsl_to_enumerator(producer.call)
          end
        else
          lambda do
            if producer.respond_to?(:rewind)
              begin
                producer.rewind
              rescue StandardError
                # Keep default Enumerable semantics when rewind is unavailable at runtime.
              end
            end
            __dsl_to_enumerator(producer)
          end
        end
      end
      private_class_method :__dsl_factory_for

      def self.__dsl_to_enumerator(value)
        unless value.respond_to?(:each)
          raise ArgumentError, "data pipeline source must respond to #each"
        end

        value.to_enum
      end
      private_class_method :__dsl_to_enumerator

      class Pipeline
        include Enumerable

        def initialize(factory)
          @factory = factory
        end

        def each
          enum = @factory.call
          return enum unless block_given?

          enum.each { |item| yield item }
        end

        def map(&block)
          raise ArgumentError, "pipeline map requires a block" unless block_given?

          self.class.new(lambda {
            upstream = @factory.call
            Enumerator.new do |y|
              index = 0
              upstream.each do |item|
                y << __dsl_call_with_context(block, item, index, "pipeline map")
                index += 1
              end
            end
          })
        end

        def filter(&block)
          raise ArgumentError, "pipeline filter requires a block" unless block_given?

          self.class.new(lambda {
            upstream = @factory.call
            Enumerator.new do |y|
              index = 0
              upstream.each do |item|
                y << item if __dsl_call_with_context(block, item, index, "pipeline filter")
                index += 1
              end
            end
          })
        end

        def batch(size, drop_last: false)
          batch_size = size.to_i
          raise ArgumentError, "pipeline batch size must be positive" if batch_size <= 0

          self.class.new(lambda {
            upstream = @factory.call
            Enumerator.new do |y|
              chunk = []
              upstream.each do |item|
                chunk << item
                if chunk.length == batch_size
                  y << chunk
                  chunk = []
                end
              end
              y << chunk unless drop_last || chunk.empty?
            end
          })
        end

        def take(count)
          limit = count.to_i
          raise ArgumentError, "pipeline take count must be non-negative" if limit.negative?

          self.class.new(lambda {
            upstream = @factory.call
            Enumerator.new do |y|
              seen = 0
              while seen < limit
                begin
                  y << upstream.next
                  seen += 1
                rescue StopIteration
                  break
                end
              end
            end
          })
        end

        def repeat(times = nil)
          if times.nil?
            self.class.new(lambda {
              Enumerator.new do |y|
                loop do
                  upstream = @factory.call
                  produced = false
                  upstream.each do |item|
                    produced = true
                    y << item
                  end
                  break unless produced
                end
              end
            })
          else
            cycles = times.to_i
            raise ArgumentError, "pipeline repeat count must be non-negative" if cycles.negative?

            self.class.new(lambda {
              Enumerator.new do |y|
                cycles.times do
                  @factory.call.each do |item|
                    y << item
                  end
                end
              end
            })
          end
        end

        def shuffle(seed: nil, random: nil)
          if !seed.nil? && !random.nil?
            raise ArgumentError, "pipeline shuffle accepts either seed: or random:, not both"
          end

          self.class.new(lambda {
            items = @factory.call.to_a
            rng = if !random.nil?
              random
            elsif !seed.nil?
              Random.new(seed.to_i)
            else
              Random.new
            end
            items.shuffle(random: rng).to_enum
          })
        end

        def prefetch(size = 1)
          prefetch_size = size.to_i
          raise ArgumentError, "pipeline prefetch size must be positive" if prefetch_size <= 0

          self.class.new(lambda {
            upstream = @factory.call
            Enumerator.new do |y|
              buffer = []

              prefetch_size.times do
                begin
                  buffer << upstream.next
                rescue StopIteration
                  break
                end
              end

              until buffer.empty?
                y << buffer.shift
                begin
                  buffer << upstream.next
                rescue StopIteration
                  # Exhausted upstream; continue draining buffer.
                end
              end
            end
          })
        end

        private

        def __dsl_call_with_context(callable, item, index, label)
          values = {
            item: item,
            index: index,
            pipeline: self
          }
          return callable.call(item, index) unless callable.respond_to?(:parameters)

          params = callable.parameters
          return callable.call(item, index) if params.empty?

          args = __dsl_build_positional_args(
            params,
            values,
            [[:item, item], [:index, index], [:pipeline, self]],
            label
          )
          kwargs = __dsl_build_keyword_args(params, values, label)
          return callable.call(*args) if kwargs.empty?

          callable.call(*args, **kwargs)
        end

        def __dsl_build_positional_args(params, values, fallback_pairs, label)
          queue = fallback_pairs.dup
          args = []
          params.each do |type, name|
            next unless type == :req || type == :opt

            if !name.nil? && values.key?(name)
              args << values.fetch(name)
              queue.reject! { |key, _value| key == name }
              next
            end

            if queue.empty?
              raise ArgumentError, "#{label} has unsupported required positional argument: #{name.inspect}" if type == :req
              break
            end

            _key, value = queue.shift
            args << value
          end
          args
        end

        def __dsl_build_keyword_args(params, values, label)
          return values.dup if params.any? { |type, _name| type == :keyrest }

          required_keys = params.each_with_object([]) do |(type, name), out|
            out << name if type == :keyreq
          end
          missing = required_keys.reject { |name| values.key?(name) }
          unless missing.empty?
            raise ArgumentError, "#{label} requires unsupported keyword argument(s): #{missing.map(&:inspect).join(", ")}"
          end

          accepted_keys = params.each_with_object([]) do |(type, name), out|
            out << name if type == :key || type == :keyreq
          end

          values.each_with_object({}) do |(name, value), out|
            out[name] = value if accepted_keys.include?(name)
          end
        end
      end
    end
  end
end
