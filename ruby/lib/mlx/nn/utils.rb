# frozen_string_literal: true

module MLX
  module NN
    module Utils
      module_function

      def value_and_grad(model, fn)
        inner_fn = lambda do |params, *args, **kwargs|
          model.update(params)
          fn.call(*args, **kwargs)
        end
        value_grad_fn = MLX::Core.value_and_grad(inner_fn)

        lambda do |*args, **kwargs|
          params = model.trainable_parameters
          if MLX::Utils.tree_flatten(params).empty?
            [fn.call(*args, **kwargs), {}]
          else
            value_grad_fn.call(params, *args, **kwargs)
          end
        end
      end

      def checkpoint(module_obj, fn = nil, &block)
        callable = fn || block
        callable = module_obj if callable.nil?

        inner_fn = lambda do |params, *args, **kwargs|
          module_obj.update(params)
          callable.call(*args, **kwargs)
        end
        checkpointed_fn = MLX::Core.checkpoint(inner_fn)

        lambda do |*args, **kwargs|
          checkpointed_fn.call(module_obj.trainable_parameters, *args, **kwargs)
        end
      end

      def average_gradients(
        gradients,
        group = nil,
        all_reduce_size: 32 * 1024 * 1024,
        communication_type: nil,
        communication_stream: nil
      )
        _ = all_reduce_size
        group ||= begin
          MLX::Core.init
        rescue StandardError
          nil
        end

        world_size = group&.respond_to?(:size) ? group.size : 1
        return gradients if world_size == 1

        average_fn = lambda do |x|
          original_dtype = x.dtype
          x = x.astype(communication_type) unless communication_type.nil?
          summed = if communication_stream.nil?
            MLX::Core.all_sum(x)
          else
            MLX::Core.all_sum(x, communication_stream)
          end
          MLX::Core.divide(summed.astype(original_dtype), world_size)
        end

        MLX::Utils.tree_map(average_fn, gradients)
      rescue StandardError
        gradients
      end
    end

    class << self
      def value_and_grad(model, fn)
        Utils.value_and_grad(model, fn)
      end

      def checkpoint(module_obj, fn = nil, &block)
        Utils.checkpoint(module_obj, fn, &block)
      end

      def average_gradients(
        gradients,
        group = nil,
        all_reduce_size: nil,
        communication_type: nil,
        communication_stream: nil
      )
        Utils.average_gradients(
          gradients,
          group,
          all_reduce_size: all_reduce_size || (32 * 1024 * 1024),
          communication_type: communication_type,
          communication_stream: communication_stream
        )
      end
    end
  end
end
