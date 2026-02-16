# frozen_string_literal: true

module MLX
  module DSL
    module Tensor
      module_function

      def scatter_rows(base:, row_indices:, values:, axis: nil)
        axis = __dsl_default_scatter_axis(base, axis)
        row_indices = __dsl_to_index_array(row_indices)
        values = __dsl_to_array(values, dtype: base.dtype)

        if base.ndim != values.ndim
          raise ArgumentError, "base and values must have the same rank"
        end

        unless base.shape.each_with_index.all? { |dim, idx| idx == axis || dim == values.shape[idx] }
          raise ArgumentError, "values shape must match base shape except along axis #{axis}"
        end

        indices = __dsl_expand_indices_for_axis(
          row_indices: row_indices,
          values_shape: values.shape,
          axis: axis
        )
        MLX::Core.put_along_axis(base, indices, values, axis)
      end

      def where_labels(base:, labels:, mapping:, mode: :add_or_replace)
        mode = mode.to_sym
        unless [:add_or_replace, :replace].include?(mode)
          raise ArgumentError, "mode must be :add_or_replace or :replace"
        end

        out = base
        mapping.each do |label_value, mapped_value|
          mask = MLX::Core.equal(labels, label_value)
          mask = __dsl_expand_trailing_dims(mask, out.ndim)
          mapped = __dsl_broadcast_mapping(mapped_value, base: out, labels_ndim: labels.ndim)
          replacement = if mode == :replace
            mapped
          else
            MLX::Core.add(out, mapped)
          end
          out = MLX::Core.where(mask, replacement, out)
        end
        out
      end

      def __dsl_default_scatter_axis(base, axis)
        return axis.to_i unless axis.nil?
        return 1 if base.ndim == 3

        0
      end
      private_class_method :__dsl_default_scatter_axis

      def __dsl_to_index_array(indices)
        if indices.is_a?(MLX::Core::Array)
          return indices.astype(MLX::Core.int32)
        end

        MLX::Core.array(indices, MLX::Core.int32)
      end
      private_class_method :__dsl_to_index_array

      def __dsl_to_array(value, dtype:)
        if value.is_a?(MLX::Core::Array)
          return value.astype(dtype)
        end

        MLX::Core.array(value, dtype)
      end
      private_class_method :__dsl_to_array

      def __dsl_expand_indices_for_axis(row_indices:, values_shape:, axis:)
        if row_indices.ndim == 1
          expected = values_shape[axis]
          if row_indices.shape[0] != expected
            raise ArgumentError, "row_indices length must match values shape at axis #{axis}"
          end

          reshape = Array.new(values_shape.length, 1)
          reshape[axis] = expected
          base = MLX::Core.reshape(row_indices, reshape)
          return MLX::Core.broadcast_to(base, values_shape)
        end

        if row_indices.shape == values_shape
          return row_indices
        end

        raise ArgumentError, "row_indices must be rank-1 or match values shape"
      end
      private_class_method :__dsl_expand_indices_for_axis

      def __dsl_expand_trailing_dims(array, target_ndim)
        out = array
        while out.ndim < target_ndim
          out = MLX::Core.expand_dims(out, out.ndim)
        end
        out
      end
      private_class_method :__dsl_expand_trailing_dims

      def __dsl_broadcast_mapping(value, base:, labels_ndim:)
        mapped = __dsl_to_array(value, dtype: base.dtype)
        return mapped if mapped.shape == base.shape

        if mapped.ndim == 1 && mapped.shape[0] == base.shape[-1]
          reshape = Array.new(labels_ndim, 1) + [mapped.shape[0]]
          mapped = MLX::Core.reshape(mapped, reshape)
          return MLX::Core.broadcast_to(mapped, base.shape)
        end

        if mapped.ndim == labels_ndim && mapped.shape == base.shape[0...labels_ndim]
          mapped = __dsl_expand_trailing_dims(mapped, base.ndim)
          return MLX::Core.broadcast_to(mapped, base.shape)
        end

        MLX::Core.broadcast_to(mapped, base.shape)
      end
      private_class_method :__dsl_broadcast_mapping
    end
  end
end
