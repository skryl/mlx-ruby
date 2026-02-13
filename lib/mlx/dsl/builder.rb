# frozen_string_literal: true

module MLX
  module DSL
    class Builder
      def initialize(owner = nil)
        @owner = owner
        @collector = nil
      end

      def build(&block)
        raise ArgumentError, "builder requires a block" unless block_given?

        instance_eval(&block)
      end

      def sequential(*modules, &block)
        collected = __dsl_modules_from(modules, &block)
        push(MLX::NN::Sequential.new(*collected))
      end

      def layer(klass, *args, **kwargs)
        push(klass.new(*args, **kwargs))
      end

      def residual(module_obj = nil, &block)
        modules = __dsl_modules_from(module_obj.nil? ? [] : [module_obj], &block)
        raise ArgumentError, "residual requires at least one module" if modules.empty?

        target = if modules.length == 1
          modules[0]
        else
          MLX::NN::Sequential.new(*modules)
        end
        push(MLX::DSL::Residual.new(target))
      end

      def branch(*modules, &block)
        collected = __dsl_modules_from(modules, &block)
        raise ArgumentError, "branch requires at least one module" if collected.empty?

        push(MLX::DSL::Parallel.new(*collected))
      end

      def concat(*modules, axis: -1, &block)
        collected = __dsl_modules_from(modules, &block)
        raise ArgumentError, "concat requires at least one module" if collected.empty?

        push(MLX::DSL::Concat.new(*collected, axis: axis))
      end

      def sum(*modules, &block)
        collected = __dsl_modules_from(modules, &block)
        raise ArgumentError, "sum requires at least one module" if collected.empty?

        push(MLX::DSL::Reduce.new(*collected, mode: :sum))
      end

      def identity(*args, **kwargs)
        push(MLX::NN::Identity.new(*args, **kwargs))
      end

      def embedding(*args, **kwargs)
        push(MLX::NN::Embedding.new(*args, **kwargs))
      end

      def linear(*args, **kwargs)
        push(MLX::NN::Linear.new(*args, **kwargs))
      end

      def bilinear(*args, **kwargs)
        push(MLX::NN::Bilinear.new(*args, **kwargs))
      end

      def relu
        push(MLX::NN::ReLU.new)
      end

      def relu6
        push(MLX::NN::ReLU6.new)
      end

      def gelu(*args, **kwargs)
        push(MLX::NN::GELU.new(*args, **kwargs))
      end

      def tanh
        push(MLX::NN::Tanh.new)
      end

      def sigmoid
        push(MLX::NN::Sigmoid.new)
      end

      def dropout(*args)
        push(MLX::NN::Dropout.new(*args))
      end

      def dropout2d(*args)
        push(MLX::NN::Dropout2d.new(*args))
      end

      def dropout3d(*args)
        push(MLX::NN::Dropout3d.new(*args))
      end

      def conv1d(*args, **kwargs)
        push(MLX::NN::Conv1d.new(*args, **kwargs))
      end

      def conv2d(*args, **kwargs)
        push(MLX::NN::Conv2d.new(*args, **kwargs))
      end

      def conv3d(*args, **kwargs)
        push(MLX::NN::Conv3d.new(*args, **kwargs))
      end

      def layer_norm(*args, **kwargs)
        push(MLX::NN::LayerNorm.new(*args, **kwargs))
      end

      def rms_norm(*args, **kwargs)
        push(MLX::NN::RMSNorm.new(*args, **kwargs))
      end

      def batch_norm(*args, **kwargs)
        push(MLX::NN::BatchNorm.new(*args, **kwargs))
      end

      def instance_norm(*args, **kwargs)
        push(MLX::NN::InstanceNorm.new(*args, **kwargs))
      end

      def group_norm(*args, **kwargs)
        push(MLX::NN::GroupNorm.new(*args, **kwargs))
      end

      def max_pool2d(*args, **kwargs)
        push(MLX::NN::MaxPool2d.new(*args, **kwargs))
      end

      def avg_pool2d(*args, **kwargs)
        push(MLX::NN::AvgPool2d.new(*args, **kwargs))
      end

      def upsample(*args, **kwargs)
        push(MLX::NN::Upsample.new(*args, **kwargs))
      end

      def method_missing(name, *args, **kwargs, &block)
        if !@owner.nil? && @owner.respond_to?(name)
          @owner.public_send(name, *args, **kwargs, &block)
        else
          super
        end
      end

      def respond_to_missing?(name, include_private = false)
        (!@owner.nil? && @owner.respond_to?(name, include_private)) || super
      end

      private

      def collect_modules(&block)
        previous = @collector
        @collector = []
        returned = instance_eval(&block)
        collected = @collector.dup
        if collected.empty? && returned.is_a?(MLX::NN::Module)
          collected << returned
        end
        collected
      ensure
        @collector = previous
      end

      def push(module_obj)
        @collector << module_obj unless @collector.nil?
        module_obj
      end

      def __dsl_modules_from(existing, &block)
        out = existing.dup
        out.concat(collect_modules(&block)) if block_given?
        out
      end
    end
  end
end
