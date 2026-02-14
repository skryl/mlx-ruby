# frozen_string_literal: true

module MLX
  module DSL
    class Callable < MLX::NN::Module
      def initialize(callable = nil, &block)
        super()
        if !callable.nil? && block_given?
          raise ArgumentError, "callable layer accepts either a callable argument or block, not both"
        end

        @callable = callable.nil? ? block : callable
        unless @callable.respond_to?(:call)
          raise ArgumentError, "callable layer requires a callable argument or block"
        end
      end

      def call(*args, **kwargs)
        return @callable.call(*args) if kwargs.empty?

        @callable.call(*args, **kwargs)
      end
    end

    class Residual < MLX::NN::Module
      def initialize(module_obj)
        super()
        self.module_obj = module_obj
      end

      def call(*args, **kwargs)
        raise ArgumentError, "residual module expects at least one positional input" if args.empty?

        identity = args[0]
        transformed = module_obj.call(*args, **kwargs)
        MLX::Core.add(identity, transformed)
      end
    end

    class Parallel < MLX::NN::Module
      def initialize(*modules)
        super()
        self.layers = modules
      end

      def call(*args, **kwargs)
        layers.map do |layer|
          layer.call(*args, **kwargs)
        end
      end
    end

    class Concat < MLX::NN::Module
      def initialize(*modules, axis: -1)
        super()
        self.layers = modules
        @axis = axis
      end

      def call(*args, **kwargs)
        outputs = layers.map do |layer|
          layer.call(*args, **kwargs)
        end
        MLX::Core.concatenate(outputs, @axis)
      end
    end

    class Reduce < MLX::NN::Module
      def initialize(*modules, mode: :sum)
        super()
        self.layers = modules
        @mode = mode.to_sym
      end

      def call(*args, **kwargs)
        outputs = layers.map do |layer|
          layer.call(*args, **kwargs)
        end

        case @mode
        when :sum
          outputs.reduce do |acc, item|
            MLX::Core.add(acc, item)
          end
        else
          raise ArgumentError, "unsupported reduce mode: #{@mode.inspect}"
        end
      end
    end
  end
end
