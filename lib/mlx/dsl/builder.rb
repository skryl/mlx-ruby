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

      def layer(entry = nil, *args, **kwargs, &block)
        if !entry.nil? && block_given?
          raise ArgumentError, "layer accepts either a module entry or block, not both"
        end

        if block_given?
          return push(MLX::DSL::Callable.new(&block))
        end

        if entry.nil?
          raise ArgumentError, "layer requires a module entry or block"
        end

        if entry.is_a?(MLX::NN::Module)
          __dsl_reject_layer_constructor_args!(args, kwargs, entry.class)
          return push(entry)
        end

        if entry.is_a?(Class)
          unless entry <= MLX::NN::Module
            raise TypeError, "layer class must inherit from MLX::NN::Module"
          end
          return push(entry.new(*args, **kwargs))
        end

        if entry.respond_to?(:call)
          __dsl_reject_layer_constructor_args!(args, kwargs, entry.class)
          return push(MLX::DSL::Callable.new(entry))
        end

        raise TypeError, "layer requires an MLX::NN::Module instance, MLX::NN::Module class, callable, or block"
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

      def fn(callable = nil, &block)
        push(MLX::DSL::Callable.new(callable, &block))
      end
      alias_method :lambda_layer, :fn

      def repeat_layers(count, &block)
        entries = __dsl_collect_repeated_entries(count, &block)
        layers = entries.map { |entry| __dsl_normalize_module_entry(entry) }
        layers.each { |layer| push(layer) }
        layers
      end

      def stack(count, layer_class = nil, *args, **kwargs, &block)
        if !layer_class.nil? && block_given?
          raise ArgumentError, "stack accepts either a layer class or block, not both"
        end

        layers = if layer_class.nil?
          __dsl_collect_repeated_entries(count, &block).map { |entry| __dsl_normalize_module_entry(entry) }
        else
          __dsl_build_class_stack_layers(count, layer_class, args, kwargs)
        end
        push(MLX::NN::Sequential.new(*layers))
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

      def leaky_relu(*args)
        push(MLX::NN::LeakyReLU.new(*args))
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

      def conv_transpose1d(*args, **kwargs)
        push(MLX::NN::ConvTranspose1d.new(*args, **kwargs))
      end

      def conv_transpose2d(*args, **kwargs)
        push(MLX::NN::ConvTranspose2d.new(*args, **kwargs))
      end

      def conv_transpose3d(*args, **kwargs)
        push(MLX::NN::ConvTranspose3d.new(*args, **kwargs))
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

      def max_pool1d(*args, **kwargs)
        push(MLX::NN::MaxPool1d.new(*args, **kwargs))
      end

      def avg_pool1d(*args, **kwargs)
        push(MLX::NN::AvgPool1d.new(*args, **kwargs))
      end

      def max_pool3d(*args, **kwargs)
        push(MLX::NN::MaxPool3d.new(*args, **kwargs))
      end

      def avg_pool3d(*args, **kwargs)
        push(MLX::NN::AvgPool3d.new(*args, **kwargs))
      end

      def rnn(*args, **kwargs)
        push(MLX::NN::RNN.new(*args, **kwargs))
      end

      def gru(*args, **kwargs)
        push(MLX::NN::GRU.new(*args, **kwargs))
      end

      def lstm(*args, **kwargs)
        push(MLX::NN::LSTM.new(*args, **kwargs))
      end

      def multi_head_attention(*args, **kwargs)
        push(MLX::NN::MultiHeadAttention.new(*args, **kwargs))
      end

      def transformer_encoder_layer(*args, **kwargs)
        push(MLX::NN::TransformerEncoderLayer.new(*args, **kwargs))
      end

      def transformer_encoder(*args, **kwargs)
        push(MLX::NN::TransformerEncoder.new(*args, **kwargs))
      end

      def transformer_decoder_layer(*args, **kwargs)
        push(MLX::NN::TransformerDecoderLayer.new(*args, **kwargs))
      end

      def transformer_decoder(*args, **kwargs)
        push(MLX::NN::TransformerDecoder.new(*args, **kwargs))
      end

      def transformer(*args, **kwargs)
        push(MLX::NN::Transformer.new(*args, **kwargs))
      end

      def rope(*args, **kwargs)
        push(MLX::NN::RoPE.new(*args, **kwargs))
      end

      def sinusoidal_positional_encoding(*args, **kwargs)
        push(MLX::NN::SinusoidalPositionalEncoding.new(*args, **kwargs))
      end

      def alibi(*args, **kwargs)
        push(MLX::NN::ALiBi.new(*args, **kwargs))
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
        if collected.empty? && !returned.nil?
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
        out.map { |entry| __dsl_normalize_module_entry(entry) }
      end

      def __dsl_normalize_module_entry(entry)
        return entry if entry.is_a?(MLX::NN::Module)

        if entry.is_a?(Class)
          return entry.new if entry <= MLX::NN::Module

          raise TypeError, "builder entries must be MLX::NN::Module instances, MLX::NN::Module classes, or callables"
        end

        return MLX::DSL::Callable.new(entry) if entry.respond_to?(:call)

        raise TypeError, "builder entries must be MLX::NN::Module instances, MLX::NN::Module classes, or callables"
      end

      def __dsl_reject_layer_constructor_args!(args, kwargs, entry_type)
        return if args.empty? && kwargs.empty?

        raise ArgumentError, "layer entry #{entry_type} does not accept constructor arguments"
      end

      def __dsl_collect_repeated_entries(count, &block)
        raise ArgumentError, "repeat requires a block" unless block_given?

        repeats = count.to_i
        raise ArgumentError, "repeat count must be non-negative" if repeats.negative?

        out = []
        repeats.times do |index|
          out.concat(
            collect_modules do
              __dsl_call_repeat_block(block, index)
            end
          )
        end
        out
      end

      def __dsl_call_repeat_block(block, index)
        return instance_eval(&block) if block.arity.zero?

        block.call(index)
      end

      def __dsl_build_class_stack_layers(count, layer_class, args, kwargs)
        repeats = count.to_i
        raise ArgumentError, "stack count must be non-negative" if repeats.negative?
        unless layer_class.is_a?(Class) && layer_class <= MLX::NN::Module
          raise TypeError, "stack layer class must inherit from MLX::NN::Module"
        end

        Array.new(repeats) { layer_class.new(*args, **kwargs) }
      end
    end
  end
end
