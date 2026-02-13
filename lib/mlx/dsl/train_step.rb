# frozen_string_literal: true

module MLX
  module DSL
    module TrainStepMethods
      def train_step(optimizer:, clip_grad_norm: nil, &loss_block)
        raise ArgumentError, "train_step requires a loss block" unless block_given?

        TrainStep.new(
          self,
          optimizer: optimizer,
          clip_grad_norm: clip_grad_norm,
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

      def initialize(model, optimizer:, clip_grad_norm:, loss_block:)
        @model = model
        @optimizer = optimizer
        @clip_grad_norm = clip_grad_norm
        @hooks = Hash.new { |h, k| h[k] = [] }
        @step = 0
        @value_and_grad = MLX::NN.value_and_grad(
          model,
          lambda do |*args, **kwargs|
            loss_block.call(*args, **kwargs)
          end
        )
      end

      def on(event, &block)
        raise ArgumentError, "hook registration requires a block" unless block_given?

        @hooks[event.to_sym] << block
        self
      end

      HOOK_EVENTS.each do |event|
        define_method(event) do |&block|
          on(event, &block)
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

        @step += 1
        loss
      end

      private

      def emit(event, context)
        @hooks[event.to_sym].each do |hook|
          hook.call(context)
        end
      end
    end
  end
end
