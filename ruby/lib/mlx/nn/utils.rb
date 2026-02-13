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
          unless contains_array_leaf?(params)
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

        return MLX::Utils.tree_map(average_fn, gradients) if all_reduce_size.to_i <= 0

        flat_grads = MLX::Utils.tree_flatten(gradients)
        return gradients if flat_grads.empty?

        keys = flat_grads.map(&:first)
        arrays = flat_grads.map(&:last)
        dtypes = arrays.map(&:dtype)
        unless dtypes.all? { |dtype| dtype == dtypes.first }
          return average_gradients(
            gradients,
            group,
            all_reduce_size: 0,
            communication_type: communication_type,
            communication_stream: communication_stream
          )
        end

        itemsize = communication_type.nil? ? dtypes.first.size : communication_type.size
        sizes = arrays.map(&:size)
        shapes = arrays.map(&:shape)

        groups = []
        current_group = []
        current_size = 0
        sizes.each_with_index do |size, i|
          current_group << i
          current_size += size * itemsize
          next unless current_size >= all_reduce_size

          groups << current_group
          current_group = []
          current_size = 0
        end
        groups << current_group unless current_group.empty?

        rebuilt = []
        groups.each do |group_indices|
          flat_arrays = group_indices.map { |index| MLX::Core.reshape(arrays[index], [sizes[index]]) }
          merged = MLX::Core.concatenate(flat_arrays)
          reduced = average_fn.call(merged)

          split_points = []
          running = 0
          group_indices.each_with_index do |index, i|
            running += sizes[index]
            split_points << running if i < group_indices.length - 1
          end

          parts = if split_points.empty?
            [reduced]
          else
            MLX::Core.split(reduced, split_points, 0)
          end

          group_indices.each_with_index do |index, part_index|
            rebuilt << [keys[index], MLX::Core.reshape(parts[part_index], shapes[index])]
          end
        end

        MLX::Utils.tree_unflatten(rebuilt)
      rescue StandardError
        gradients
      end

      def contains_array_leaf?(tree)
        return true if tree.is_a?(MLX::Core::Array)
        return tree.any? { |item| contains_array_leaf?(item) } if tree.is_a?(::Array)
        return tree.any? { |_k, value| contains_array_leaf?(value) } if tree.is_a?(::Hash)

        false
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
