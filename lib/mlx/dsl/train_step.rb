# frozen_string_literal: true

module MLX
  module DSL
    module TrainStepMethods
      def train_step(optimizer:, clip_grad_norm: nil, compile: false, sync: :none, &loss_block)
        raise ArgumentError, "train_step requires a loss block" unless block_given?

        TrainStep.new(
          self,
          optimizer: optimizer,
          clip_grad_norm: clip_grad_norm,
          compile: compile,
          sync: sync,
          loss_block: loss_block
        )
      end
    end

    class TrainStep
      HOOK_EVENTS = %i[
        before_step
        after_backward
        after_step
      ].freeze

      def initialize(model, optimizer:, clip_grad_norm:, compile: false, sync: :none, loss_block:)
        @model = model
        @optimizer = optimizer
        @clip_grad_norm = clip_grad_norm
        @sync_mode = __dsl_normalize_sync(sync)
        @hooks = Hash.new { |h, k| h[k] = [] }
        @hook_order = 0
        @step = 0
        value_and_grad = MLX::NN.value_and_grad(
          model,
          lambda do |*args, **kwargs|
            loss_block.call(*args, **kwargs)
          end
        )
        @value_and_grad = __dsl_compile_callable(value_and_grad, compile)
      end

      def on(event, priority: 0, every: nil, once: false, **kwargs, &block)
        raise ArgumentError, "hook registration requires a block" unless block_given?
        condition = kwargs.delete(:if)
        condition = kwargs.delete(:condition) if condition.nil? && kwargs.key?(:condition)
        unless kwargs.empty?
          raise ArgumentError, "unsupported hook option(s): #{kwargs.keys.map(&:inspect).join(', ')}"
        end
        every_value = nil
        unless every.nil?
          every_value = every.to_i
          raise ArgumentError, "hook :every must be a positive integer" if every_value <= 0
        end

        @hooks[event.to_sym] << {
          hook: block,
          priority: priority.to_i,
          every: every_value,
          once: !!once,
          if: condition,
          fired: false,
          invocations: 0,
          order: @hook_order
        }
        @hook_order += 1
        self
      end

      HOOK_EVENTS.each do |event|
        define_method(event) do |**options, &block|
          on(event, **options, &block)
        end
      end

      def call(*args, **kwargs)
        context = {
          step: @step,
          model: @model,
          optimizer: @optimizer,
          args: args,
          kwargs: kwargs
        }
        emit(:before_step, context)

        loss, grads = @value_and_grad.call(*args, **kwargs)
        context[:loss] = loss
        context[:grads] = grads
        emit(:after_backward, context)

        if !@clip_grad_norm.nil?
          grads, total_norm = MLX::Optimizers.clip_grad_norm(grads, @clip_grad_norm)
          context[:grads] = grads
          context[:grad_norm] = total_norm
        end

        @optimizer.update(@model, grads)
        emit(:after_step, context)
        __dsl_sync_step(loss) if @sync_mode == :step

        @step += 1
        loss
      end

      private

      def emit(event, context)
        @hooks[event.to_sym].sort_by { |entry| [entry.fetch(:priority), entry.fetch(:order)] }.each do |entry|
          entry[:invocations] += 1
          if !entry[:every].nil? && ((entry[:invocations] - 1) % entry[:every]).nonzero?
            next
          end
          if entry[:once] && entry[:fired]
            next
          end
          unless __dsl_hook_condition_met?(entry[:if], context)
            next
          end

          entry.fetch(:hook).call(context)
          entry[:fired] = true if entry[:once]
        end
      end

      def __dsl_hook_condition_met?(condition, context)
        return true if condition.nil?
        return !!condition unless condition.respond_to?(:call)
        return !!condition.call unless condition.respond_to?(:parameters)

        params = condition.parameters
        return !!condition.call if params.empty?

        if params.any? { |type, _name| type == :keyrest || type == :key || type == :keyreq }
          return !!condition.call(context: context)
        end

        !!condition.call(context)
      end

      def __dsl_compile_callable(callable, compile)
        config = __dsl_compile_config(compile)
        return callable unless config[:enabled]
        unless defined?(MLX::Core) && MLX::Core.respond_to?(:compile)
          raise ArgumentError, "compile requested but MLX::Core.compile is unavailable"
        end

        MLX::Core.compile(callable, config[:inputs], config[:outputs], config[:shapeless])
      end

      def __dsl_compile_config(compile)
        case compile
        when nil, false
          { enabled: false, inputs: nil, outputs: nil, shapeless: false }
        when true
          { enabled: true, inputs: nil, outputs: nil, shapeless: false }
        when Hash
          kwargs = compile.each_with_object({}) do |(key, value), out|
            out[key.to_sym] = value
          end
          extra = kwargs.keys - %i[inputs outputs shapeless]
          unless extra.empty?
            raise ArgumentError, "unsupported compile option(s): #{extra.map(&:inspect).join(', ')}"
          end

          {
            enabled: true,
            inputs: kwargs[:inputs],
            outputs: kwargs[:outputs],
            shapeless: !!kwargs[:shapeless]
          }
        else
          raise ArgumentError, "compile must be a boolean or options hash"
        end
      end

      def __dsl_normalize_sync(sync)
        mode = sync.nil? ? :none : sync.to_sym
        return mode if %i[none step].include?(mode)

        raise ArgumentError, "train_step sync must be one of :none or :step"
      end

      def __dsl_sync_step(loss)
        return unless defined?(MLX::Core) && MLX::Core.respond_to?(:eval)

        targets = []
        targets << loss unless loss.nil?
        targets << @model.parameters if @model.respond_to?(:parameters)
        targets << @optimizer.state if @optimizer.respond_to?(:state)
        return if targets.empty?

        MLX::Core.eval(*targets)
      end
    end
  end
end
