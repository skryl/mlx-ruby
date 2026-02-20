# frozen_string_literal: true

module MLX
  module GraphIR
    module_function

    def reduce_axes_from_arguments(arguments)
      unless arguments.is_a?(Array) && arguments.length >= 2
        raise ArgumentError, "[graph_ir_to_onnx_stub] Reduce arguments must include reduction code and axes"
      end
      normalize_integer_vector(arguments[1], "Reduce axes")
    end
    private_class_method :reduce_axes_from_arguments

    def infer_reduce_keepdims_shape(input_shape, axes)
      return nil if input_shape.nil?

      shape = normalize_integer_vector(input_shape, "Reduce input shape")
      rank = shape.length
      normalized_axes = normalize_integer_vector(axes, "Reduce axes").map do |axis|
        normalize_axis(axis, rank, "Reduce axis")
      end.uniq

      normalized_axes.each { |axis| shape[axis] = 1 }
      shape
    end
    private_class_method :infer_reduce_keepdims_shape

    def as_type_target_dtype(arguments, outputs, known_dtypes)
      if arguments.is_a?(Array) && !arguments.empty?
        target = arguments.first
        unless target.is_a?(String) && ONNX_DTYPE_MAP.key?(target)
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported AsType arguments #{arguments.inspect}; expected first argument to be dtype String"
        end
        return target
      end

      candidates = outputs.map { |name| known_dtypes[name] }.compact.uniq
      return candidates.first if candidates.length == 1
      if candidates.length > 1
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported AsType with inconsistent output dtypes #{candidates.inspect}"
      end

      raise NotImplementedError,
            "[graph_ir_to_onnx_stub] unsupported AsType without target dtype argument"
    end
    private_class_method :as_type_target_dtype

    def equal_nan_from_arguments(arguments)
      return false if arguments.nil? || arguments.empty?
      unless arguments.is_a?(Array) && arguments.length == 1 && (arguments[0] == true || arguments[0] == false)
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Equal arguments #{arguments.inspect}; expected [equal_nan]"
      end

      arguments[0]
    end
    private_class_method :equal_nan_from_arguments

    def infer_logsumexp_axes(input_shape, output_shape)
      input = normalize_integer_vector(input_shape, "LogSumExp input shape")
      if output_shape
        output = normalize_integer_vector(output_shape, "LogSumExp output shape")
        if output.length == input.length
          axes = []
          input.each_with_index do |dim, index|
            out_dim = output[index]
            if out_dim == 1 && dim != 1
              axes << index
            elsif out_dim != dim
              raise NotImplementedError,
                    "[graph_ir_to_onnx_stub] unsupported LogSumExp output shape #{output.inspect} for input #{input.inspect}"
            end
          end
          return [input.length - 1] if axes.empty?

          return axes
        end

        if output.length == input.length - 1
          return [input.length - 1]
        end
      end

      [input.length - 1]
    end
    private_class_method :infer_logsumexp_axes

    def pad_axes_and_sizes_from_arguments(arguments, input_shape)
      unless arguments.is_a?(Array) && arguments.length >= 3
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Pad arguments #{arguments.inspect}; expected [axes, low, high]"
      end

      axes = normalize_integer_vector(arguments[0], "Pad axes")
      low = normalize_integer_vector(arguments[1], "Pad low")
      high = normalize_integer_vector(arguments[2], "Pad high")
      unless axes.length == low.length && low.length == high.length
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] Pad axes/low/high lengths must match: " \
              "#{axes.length}/#{low.length}/#{high.length}"
      end
      if low.any?(&:negative?) || high.any?(&:negative?)
        raise NotImplementedError, "[graph_ir_to_onnx_stub] unsupported Pad with negative padding"
      end

      rank = normalize_integer_vector(input_shape, "Pad input shape").length
      normalized_axes = axes.map { |axis| normalize_axis(axis, rank, "Pad axis") }
      if normalized_axes.uniq.length != normalized_axes.length
        raise ArgumentError, "[graph_ir_to_onnx_stub] Pad axes must not contain duplicates"
      end

      [normalized_axes, low, high]
    end
    private_class_method :pad_axes_and_sizes_from_arguments

    def infer_pad_output_shape(input_shape, pads_begin, pads_end)
      return nil if input_shape.nil?

      shape = normalize_integer_vector(input_shape, "Pad input shape")
      low = normalize_integer_vector(pads_begin, "Pad low")
      high = normalize_integer_vector(pads_end, "Pad high")
      unless low.length == shape.length && high.length == shape.length
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] Pad low/high ranks must match input rank #{shape.length}"
      end
      shape.each_with_index.map { |dim, index| dim + low[index] + high[index] }
    end
    private_class_method :infer_pad_output_shape

    def scan_arguments(arguments)
      unless arguments.is_a?(Array) && arguments.length >= 4
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Scan arguments #{arguments.inspect}; expected [reduce_type, axis, reverse, inclusive]"
      end

      reduce_type = arguments[0]
      axis = arguments[1]
      reverse = arguments[2]
      inclusive = arguments[3]
      unless reduce_type.is_a?(Integer) && axis.is_a?(Integer)
        raise TypeError, "[graph_ir_to_onnx_stub] Scan reduce_type/axis must be Integer"
      end
      unless reverse == true || reverse == false
        raise TypeError, "[graph_ir_to_onnx_stub] Scan reverse must be boolean"
      end
      unless inclusive == true || inclusive == false
        raise TypeError, "[graph_ir_to_onnx_stub] Scan inclusive must be boolean"
      end

      [reduce_type, axis, reverse, inclusive]
    end
    private_class_method :scan_arguments

    def argreduce_mode_axis(arguments)
      unless arguments.is_a?(Array) && arguments.length >= 2
        raise ArgumentError, "[graph_ir_to_onnx_stub] ArgReduce arguments must include [mode, axis]"
      end

      mode = arguments[0]
      axis = arguments[1]
      unless mode.is_a?(Integer) && axis.is_a?(Integer)
        raise TypeError, "[graph_ir_to_onnx_stub] ArgReduce mode/axis must be Integer"
      end
      unless ARG_REDUCE_CODE_TO_ONNX_OP.key?(mode)
        raise NotImplementedError, "[graph_ir_to_onnx_stub] unsupported ArgReduce code #{mode.inspect}"
      end

      [mode, axis]
    end
    private_class_method :argreduce_mode_axis

    def infer_argreduce_keepdims_shape(input_shape, axis)
      return nil if input_shape.nil?

      shape = normalize_integer_vector(input_shape, "ArgReduce input shape")
      axis_index = normalize_axis(axis, shape.length, "ArgReduce axis")
      shape[axis_index] = 1
      shape
    end
    private_class_method :infer_argreduce_keepdims_shape

    def infer_matmul_output_shape(lhs_shape, rhs_shape)
      return nil if lhs_shape.nil? || rhs_shape.nil?

      lhs = normalize_integer_vector(lhs_shape, "Matmul lhs shape")
      rhs = normalize_integer_vector(rhs_shape, "Matmul rhs shape")
      return nil if lhs.empty? || rhs.empty?

      lhs_was_1d = lhs.length == 1
      rhs_was_1d = rhs.length == 1
      lhs_matrix = lhs_was_1d ? [1, lhs[0]] : lhs[-2..]
      rhs_matrix = rhs_was_1d ? [rhs[0], 1] : rhs[-2..]
      return nil unless lhs_matrix[1] == rhs_matrix[0]

      lhs_batch = lhs_was_1d ? [] : lhs[0...-2]
      rhs_batch = rhs_was_1d ? [] : rhs[0...-2]
      batch = infer_elementwise_output_shape(lhs_batch, rhs_batch)
      return nil if batch.nil?

      out = [*batch, lhs_matrix[0], rhs_matrix[1]]
      out.delete_at(batch.length) if lhs_was_1d
      out.delete_at(-1) if rhs_was_1d
      out
    end
    private_class_method :infer_matmul_output_shape

    def infer_convolution_output_shape(
      input_shape,
      weight_shape,
      strides,
      padding_low,
      padding_high,
      kernel_dilation,
      groups
    )
      return nil if input_shape.nil? || weight_shape.nil?

      input = normalize_integer_vector(input_shape, "Convolution input shape")
      weight = normalize_integer_vector(weight_shape, "Convolution weight shape")
      stride_values = normalize_integer_vector(strides, "Convolution strides")
      pad_low = normalize_integer_vector(padding_low, "Convolution padding_low")
      pad_high = normalize_integer_vector(padding_high, "Convolution padding_high")
      dilation_values = normalize_integer_vector(kernel_dilation, "Convolution kernel_dilation")

      spatial_rank = stride_values.length
      expected_rank = spatial_rank + 2
      return nil unless input.length == expected_rank && weight.length == expected_rank
      return nil unless [pad_low.length, pad_high.length, dilation_values.length].all? { |length| length == spatial_rank }

      batch = input[0]
      input_channels = input[-1]
      output_channels = weight[0]
      kernel_shape = weight[1..-2]
      weight_input_channels = weight[-1]
      return nil unless input_channels == weight_input_channels * groups

      output_spatial = spatial_rank.times.map do |axis|
        input_dim = input[axis + 1]
        kernel_dim = kernel_shape[axis]
        dilation = dilation_values[axis]
        stride = stride_values[axis]
        low = pad_low[axis]
        high = pad_high[axis]

        return nil if kernel_dim <= 0
        effective_kernel = dilation * (kernel_dim - 1) + 1
        numerator = input_dim + low + high - effective_kernel
        return nil if numerator < 0

        (numerator / stride) + 1
      end

      [batch, *output_spatial, output_channels]
    end
    private_class_method :infer_convolution_output_shape

    def infer_convolution_transpose_output_shape(
      input_shape,
      weight_shape,
      strides,
      pads_begin,
      pads_end,
      kernel_dilation,
      output_padding,
      groups
    )
      return nil if input_shape.nil? || weight_shape.nil?

      input = normalize_integer_vector(input_shape, "ConvolutionTranspose input shape")
      weight = normalize_integer_vector(weight_shape, "ConvolutionTranspose weight shape")
      stride_values = normalize_integer_vector(strides, "ConvolutionTranspose strides")
      pad_begin = normalize_integer_vector(pads_begin, "ConvolutionTranspose pads_begin")
      pad_end = normalize_integer_vector(pads_end, "ConvolutionTranspose pads_end")
      dilation_values = normalize_integer_vector(kernel_dilation, "ConvolutionTranspose dilations")
      out_padding_values = normalize_integer_vector(output_padding, "ConvolutionTranspose output_padding")

      spatial_rank = stride_values.length
      expected_rank = spatial_rank + 2
      return nil unless input.length == expected_rank && weight.length == expected_rank
      return nil unless [pad_begin.length, pad_end.length, dilation_values.length, out_padding_values.length].all? do |length|
        length == spatial_rank
      end

      batch = input[0]
      input_channels = input[-1]
      output_channels = weight[0]
      kernel_shape = weight[1..-2]
      weight_input_channels = weight[-1]
      return nil unless input_channels == weight_input_channels * groups

      output_spatial = spatial_rank.times.map do |axis|
        input_dim = input[axis + 1]
        kernel_dim = kernel_shape[axis]
        dilation = dilation_values[axis]
        stride = stride_values[axis]
        low = pad_begin[axis]
        high = pad_end[axis]
        out_padding = out_padding_values[axis]

        return nil if kernel_dim <= 0
        effective_kernel = dilation * (kernel_dim - 1) + 1
        dim = stride * (input_dim - 1) + out_padding + effective_kernel - low - high
        return nil if dim < 0

        dim
      end

      [batch, *output_spatial, output_channels]
    end
    private_class_method :infer_convolution_transpose_output_shape

    def infer_gather_output_shape(data_shape, indices_shape, axis)
      return nil if data_shape.nil? || indices_shape.nil?

      data = normalize_integer_vector(data_shape, "Gather data shape")
      indices = normalize_integer_vector(indices_shape, "Gather indices shape")
      [*data[0...axis], *indices, *(data[(axis + 1)..] || [])]
    end
    private_class_method :infer_gather_output_shape

    def slice_vectors_from_arguments(arguments)
      unless arguments.is_a?(Array) && arguments.length >= 2
        raise ArgumentError, "[graph_ir_to_onnx_stub] Slice arguments must include starts and ends"
      end

      starts = normalize_integer_vector(arguments[0], "Slice starts")
      ends = normalize_integer_vector(arguments[1], "Slice ends")
      steps = if arguments.length >= 3
        normalize_integer_vector(arguments[2], "Slice steps")
      else
        Array.new(starts.length, 1)
      end

      unless starts.length == ends.length && starts.length == steps.length
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] Slice starts/ends/steps lengths must match: " \
              "#{starts.length}/#{ends.length}/#{steps.length}"
      end
      if steps.any?(&:zero?)
        raise ArgumentError, "[graph_ir_to_onnx_stub] Slice steps must not contain zero"
      end

      axes = (0...starts.length).to_a
      [starts, ends, axes, steps]
    end
    private_class_method :slice_vectors_from_arguments

    def infer_slice_output_shape(input_shape, starts, ends, axes, steps)
      return nil if input_shape.nil?

      shape = normalize_integer_vector(input_shape, "Slice input shape")
      out_shape = shape.dup

      axes.each_with_index do |axis, index|
        axis_index = normalize_axis(axis, shape.length, "Slice axis")
        dim = shape[axis_index]
        start_v = normalize_slice_index(starts[index], dim)
        end_v = normalize_slice_index(ends[index], dim)
        step_v = steps[index]
        return nil if step_v <= 0

        out_shape[axis_index] = if end_v <= start_v
          0
        else
          ((end_v - start_v - 1) / step_v) + 1
        end
      end

      out_shape
    end
    private_class_method :infer_slice_output_shape

    def split_axis_and_lengths(arguments, input_shape, output_count)
      unless arguments.is_a?(Array) && arguments.length >= 2
        raise ArgumentError, "[graph_ir_to_onnx_stub] Split arguments must include split spec and axis"
      end

      spec = normalize_integer_vector(arguments[0], "Split spec")
      axis = arguments[1]
      unless axis.is_a?(Integer)
        raise TypeError, "[graph_ir_to_onnx_stub] Split axis must be an Integer"
      end
      unless output_count.is_a?(Integer) && output_count > 0
        raise ArgumentError, "[graph_ir_to_onnx_stub] Split must have at least one output"
      end

      unless input_shape
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Split without known input shape"
      end
      data_shape = normalize_integer_vector(input_shape, "Split input shape")
      axis_index = normalize_axis(axis, data_shape.length, "Split axis")
      dim = data_shape[axis_index]

      lengths = if spec.length == 1 && spec.first == output_count
        parts = spec.first
        if parts <= 0
          raise ArgumentError, "[graph_ir_to_onnx_stub] Split parts must be positive"
        end
        quotient, remainder = dim.divmod(parts)
        unless remainder.zero?
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported uneven equal Split: dim #{dim} not divisible by #{parts}"
        end
        Array.new(parts, quotient)
      elsif spec.length == output_count - 1
        split_lengths_from_indices(spec, dim)
      elsif spec.length == output_count
        spec
      else
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Split spec #{spec.inspect} for #{output_count} outputs"
      end

      unless lengths.length == output_count
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] Split lengths count #{lengths.length} does not match outputs #{output_count}"
      end
      if lengths.any?(&:negative?)
        raise ArgumentError, "[graph_ir_to_onnx_stub] Split lengths must be non-negative"
      end
      unless lengths.sum == dim
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Split lengths #{lengths.inspect}; expected sum #{dim}"
      end

      [axis_index, lengths]
    end
    private_class_method :split_axis_and_lengths

    def split_lengths_from_indices(indices, dim)
      prev = 0
      lengths = indices.map do |index|
        unless index.is_a?(Integer)
          raise TypeError, "[graph_ir_to_onnx_stub] Split index boundaries must be Integer"
        end
        value = index
        value += dim if value.negative?
        if value < prev || value > dim
          raise ArgumentError,
                "[graph_ir_to_onnx_stub] Split boundary #{index} is out of range or not non-decreasing for dim #{dim}"
        end
        len = value - prev
        prev = value
        len
      end
      lengths << (dim - prev)
      lengths
    end
    private_class_method :split_lengths_from_indices

    def infer_split_output_shapes(input_shape, axis, lengths)
      return nil if input_shape.nil?

      shape = normalize_integer_vector(input_shape, "Split input shape")
      lengths.map do |length|
        out = shape.dup
        out[axis] = length
        out
      end
    end
    private_class_method :infer_split_output_shapes

    def infer_concatenate_output_shape(input_shapes, axis)
      return nil if input_shapes.any?(&:nil?)

      shapes = input_shapes.map { |shape| normalize_integer_vector(shape, "Concatenate input shape") }
      return nil if shapes.empty?

      rank = shapes.first.length
      return nil unless shapes.all? { |shape| shape.length == rank }
      axis_index = normalize_axis(axis, rank, "Concatenate axis")

      out = shapes.first.dup
      out[axis_index] = 0
      shapes.each do |shape|
        rank.times do |dim_index|
          next if dim_index == axis_index
          return nil unless shape[dim_index] == out[dim_index]
        end
        out[axis_index] += shape[axis_index]
      end
      out
    end
    private_class_method :infer_concatenate_output_shape

    def infer_squeeze_output_shape(input_shape, axes)
      return nil if input_shape.nil?

      shape = normalize_integer_vector(input_shape, "Squeeze input shape")
      rank = shape.length
      normalized_axes = normalize_integer_vector(axes, "Squeeze axes")
        .map { |axis| normalize_axis(axis, rank, "Squeeze axis") }
        .uniq

      normalized_axes.sort.reverse_each do |axis_index|
        dim = shape[axis_index]
        unless dim == 1
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Squeeze axis #{axis_index} for dim #{dim}; expected dimension 1"
        end
        shape.delete_at(axis_index)
      end
      shape
    end
    private_class_method :infer_squeeze_output_shape

    def infer_unsqueeze_output_shape(input_shape, axes)
      return nil if input_shape.nil?

      shape = normalize_integer_vector(input_shape, "ExpandDims input shape")
      axes_vector = normalize_integer_vector(axes, "ExpandDims axes")
      output_rank = shape.length + axes_vector.length
      normalized_axes = axes_vector.map do |axis|
        value = axis
        value += output_rank if value.negative?
        unless value.between?(0, output_rank - 1)
          raise ArgumentError,
                "[graph_ir_to_onnx_stub] ExpandDims axis #{axis} is out of bounds for output rank #{output_rank}"
        end
        value
      end
      if normalized_axes.uniq.length != normalized_axes.length
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] ExpandDims axes #{axes.inspect} must not contain duplicates"
      end

      out = shape.dup
      normalized_axes.sort.each do |axis_index|
        out.insert(axis_index, 1)
      end
      out
    end
    private_class_method :infer_unsqueeze_output_shape

    def infer_elementwise_output_shape(lhs_shape, rhs_shape)
      return nil if lhs_shape.nil? || rhs_shape.nil?

      lhs = normalize_integer_vector(lhs_shape, "lhs shape")
      rhs = normalize_integer_vector(rhs_shape, "rhs shape")
      max_rank = [lhs.length, rhs.length].max
      lhs_aligned = Array.new(max_rank - lhs.length, 1) + lhs
      rhs_aligned = Array.new(max_rank - rhs.length, 1) + rhs

      out = lhs_aligned.zip(rhs_aligned).map do |left, right|
        if left == right
          left
        elsif left == 1
          right
        elsif right == 1
          left
        else
          return nil
        end
      end
      out
    end
    private_class_method :infer_elementwise_output_shape

    def normalize_slice_index(value, dim)
      index = value
      index += dim if index.negative?
      index = 0 if index < 0
      index = dim if index > dim
      index
    end
    private_class_method :normalize_slice_index

    def gather_reorder_permutation(data_rank, indices_rank, axis)
      indices = (axis...(axis + indices_rank)).to_a
      prefix = (0...axis).to_a
      suffix_start = axis + indices_rank
      suffix_end = data_rank + indices_rank - 2
      suffix = suffix_start > suffix_end ? [] : (suffix_start..suffix_end).to_a
      [*indices, *prefix, *suffix]
    end
    private_class_method :gather_reorder_permutation

    def identity_permutation?(perm)
      perm.each_with_index.all? { |value, index| value == index }
    end
    private_class_method :identity_permutation?

    def convolution_data_permutations(spatial_rank)
      rank = spatial_rank + 2
      to_onnx = [0, rank - 1, *(1...(rank - 1)).to_a]
      from_onnx = [0, *((2...rank).to_a), 1]
      [to_onnx, from_onnx]
    end
    private_class_method :convolution_data_permutations

    def convolution_weight_permutation(spatial_rank)
      rank = spatial_rank + 2
      [0, rank - 1, *(1...(rank - 1)).to_a]
    end
    private_class_method :convolution_weight_permutation

    def convolution_transpose_weight_permutation(spatial_rank)
      rank = spatial_rank + 2
      [rank - 1, 0, *(1...(rank - 1)).to_a]
    end
    private_class_method :convolution_transpose_weight_permutation

    def permute_shape(shape, perm, label)
      dims = normalize_integer_vector(shape, label)
      unless perm.length == dims.length && perm.sort == (0...dims.length).to_a
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] invalid permutation #{perm.inspect} for #{label} rank #{dims.length}"
      end

      perm.map { |axis| dims[axis] }
    end
    private_class_method :permute_shape

    def integer_vector_argument(arguments, op_name)
      candidate = if arguments.is_a?(Array)
        arguments.find do |value|
          next false unless value.is_a?(Array)

          begin
            normalize_integer_vector(value, "#{op_name} argument")
            true
          rescue TypeError, RangeError
            false
          end
        end
      end
      unless candidate
        raise ArgumentError, "[graph_ir_to_onnx_stub] #{op_name} requires an integer-vector argument"
      end

      normalize_integer_vector(candidate, "#{op_name} argument")
    end
    private_class_method :integer_vector_argument

    def integer_like_numeric?(value)
      return true if value.is_a?(Integer)
      return false unless value.is_a?(Numeric)

      float = value.to_f
      float.finite? && float == float.truncate
    end
    private_class_method :integer_like_numeric?

    def normalized_integer_scalar(value, label)
      unless value.is_a?(Numeric)
        raise TypeError, "[graph_ir_to_onnx_stub] #{label} must be an Integer"
      end

      raw = if value.is_a?(Integer)
        value
      else
        float = value.to_f
        unless float.finite? && float == float.truncate
          raise TypeError, "[graph_ir_to_onnx_stub] #{label} must be an Integer"
        end
        float.to_i
      end

      int64_min = -(1 << 63)
      int64_max = (1 << 63) - 1
      return raw if raw.between?(int64_min, int64_max)

      uint64_max = (1 << 64) - 1
      uint64_modulus = 1 << 64
      if raw.positive? && raw <= uint64_max
        wrapped = raw - uint64_modulus
        return wrapped if wrapped.between?(int64_min, int64_max)
      end

      raise RangeError,
            "[graph_ir_to_onnx_stub] #{label} #{value.inspect} is outside supported signed 64-bit range"
    end
    private_class_method :normalized_integer_scalar

    def normalize_integer_vector(value, label)
      case value
      when Integer
        [normalized_integer_scalar(value, label)]
      when Numeric
        [normalized_integer_scalar(value, label)]
      when Array
        value.map { |item| normalized_integer_scalar(item, label) }
      else
        raise TypeError, "[graph_ir_to_onnx_stub] #{label} must be an Integer or Array of Integer"
      end
    end
    private_class_method :normalize_integer_vector

    def flatten_shape_from_arguments(arguments, input_shape)
      start_axis, end_axis = arguments
      shape = normalize_integer_vector(input_shape, "Flatten input shape")
      rank = shape.length
      raise ArgumentError, "[graph_ir_to_onnx_stub] Flatten input shape must have rank >= 1" if rank <= 0

      start_index = normalize_axis(start_axis, rank, "Flatten start_axis")
      end_index = normalize_axis(end_axis, rank, "Flatten end_axis")
      if end_index < start_index
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Flatten axis range #{arguments.inspect} for rank #{rank}"
      end

      prefix = shape[0...start_index]
      middle = shape[start_index..end_index]
      suffix = shape[(end_index + 1)..] || []
      [*prefix, middle.reduce(1, :*), *suffix]
    end
    private_class_method :flatten_shape_from_arguments

    def unflatten_shape_from_arguments(arguments, input_shape)
      unless arguments.is_a?(Array) && arguments.length >= 2
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] Unflatten arguments must include [axis, shape]"
      end

      axis = arguments[0]
      target_shape = normalize_integer_vector(arguments[1], "Unflatten shape")
      source_shape = normalize_integer_vector(input_shape, "Unflatten input shape")
      source_rank = source_shape.length
      axis_index = normalize_axis(axis, source_rank, "Unflatten axis")
      source_dim = source_shape[axis_index]

      negative_indices = []
      known_product = 1
      target_shape.each_with_index do |dim, index|
        unless dim.is_a?(Integer)
          raise TypeError, "[graph_ir_to_onnx_stub] Unflatten shape must contain only Integer values"
        end
        if dim == -1
          negative_indices << index
          next
        end
        if dim <= 0
          raise ArgumentError, "[graph_ir_to_onnx_stub] Unflatten shape values must be positive or -1"
        end
        known_product *= dim
      end

      if negative_indices.length > 1
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Unflatten shape #{target_shape.inspect}; at most one -1 is allowed"
      end

      resolved_shape = target_shape.dup
      if negative_indices.length == 1
        unknown_index = negative_indices.first
        if known_product <= 0 || source_dim % known_product != 0
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Unflatten shape #{target_shape.inspect} for source dim #{source_dim}"
        end
        resolved_shape[unknown_index] = source_dim / known_product
      elsif known_product != source_dim
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Unflatten shape #{target_shape.inspect}; " \
              "product #{known_product} must match source dim #{source_dim}"
      end

      [*source_shape[0...axis_index], *resolved_shape, *(source_shape[(axis_index + 1)..] || [])]
    end
    private_class_method :unflatten_shape_from_arguments

    def arange_arguments(arguments)
      unless arguments.is_a?(Array) && arguments.length >= 3
        raise ArgumentError, "[graph_ir_to_onnx_stub] Arange arguments must include [start, stop, step]"
      end

      start = arguments[0]
      stop = arguments[1]
      step = arguments[2]
      unless [start, stop, step].all? { |value| value.is_a?(Numeric) }
        raise TypeError, "[graph_ir_to_onnx_stub] Arange start/stop/step must be Numeric"
      end

      integral = [start, stop, step].all? { |value| integer_like_numeric?(value) }
      if integral
        start = normalized_integer_scalar(start, "Arange start")
        stop = normalized_integer_scalar(stop, "Arange stop")
        step = normalized_integer_scalar(step, "Arange step")
        dtype = "int64"
      else
        start = start.to_f
        stop = stop.to_f
        step = step.to_f
        unless start.finite? && stop.finite? && step.finite?
          raise TypeError, "[graph_ir_to_onnx_stub] Arange start/stop/step must be finite Numeric values"
        end
        dtype = "float32"
      end
      raise ArgumentError, "[graph_ir_to_onnx_stub] Arange step must not be zero" if step.zero?

      [start, stop, step, dtype]
    end
    private_class_method :arange_arguments

    def arange_values(start, stop, step)
      values = []
      current = start
      if step.positive?
        while current < stop
          values << current
          current += step
        end
      else
        while current > stop
          values << current
          current += step
        end
      end
      values
    end
    private_class_method :arange_values

    def addmm_alpha_beta(arguments)
      return [1.0, 1.0] if arguments.nil? || arguments.empty?

      unless arguments.is_a?(Array) && arguments.length >= 2
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] AddMM arguments must include [alpha, beta]"
      end
      alpha = arguments[0]
      beta = arguments[1]
      unless alpha.is_a?(Numeric) && beta.is_a?(Numeric)
        raise TypeError, "[graph_ir_to_onnx_stub] AddMM alpha/beta must be Numeric"
      end

      [alpha.to_f, beta.to_f]
    end
    private_class_method :addmm_alpha_beta

    def sqrt_is_reciprocal?(arguments)
      return false if arguments.nil? || arguments.empty?
      return false unless arguments.is_a?(Array)

      value = arguments.first
      return value if value == true || value == false

      raise TypeError, "[graph_ir_to_onnx_stub] Sqrt reciprocal flag must be boolean when present"
    end
    private_class_method :sqrt_is_reciprocal?

    def asstrided_arguments(arguments)
      unless arguments.is_a?(Array) && arguments.length >= 3
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] AsStrided arguments must include [shape, strides, offset]"
      end

      output_shape = normalize_integer_vector(arguments[0], "AsStrided shape")
      strides = normalize_integer_vector(arguments[1], "AsStrided strides")
      offset = arguments[2]
      unless offset.is_a?(Integer)
        raise TypeError, "[graph_ir_to_onnx_stub] AsStrided offset must be an Integer"
      end
      unless output_shape.length == strides.length
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] AsStrided shape/strides length mismatch: " \
              "#{output_shape.length}/#{strides.length}"
      end
      if output_shape.any?(&:negative?)
        raise ArgumentError, "[graph_ir_to_onnx_stub] AsStrided shape values must be non-negative"
      end

      [output_shape, strides, offset]
    end
    private_class_method :asstrided_arguments

    def tensor_size_from_shape(shape, label)
      dims = normalize_integer_vector(shape, label)
      dims.empty? ? 1 : dims.reduce(1, :*)
    end
    private_class_method :tensor_size_from_shape

    def asstrided_linear_indices(output_shape, strides, offset, input_size)
      return [] if output_shape.any?(&:zero?)

      total = output_shape.empty? ? 1 : output_shape.reduce(1, :*)
      indices = []
      total.times do |linear_index|
        remainder = linear_index
        source_index = offset

        (output_shape.length - 1).downto(0) do |axis|
          dim = output_shape[axis]
          coord = remainder % dim
          remainder /= dim
          source_index += coord * strides[axis]
        end

        unless source_index.between?(0, input_size - 1)
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported AsStrided index #{source_index} " \
                "out of bounds for input size #{input_size}"
        end
        indices << source_index
      end

      indices
    end
    private_class_method :asstrided_linear_indices

    def normalize_axis(axis, rank, label)
      index = normalized_integer_scalar(axis, label)
      index += rank if index.negative?
      unless index.between?(0, rank - 1)
        raise ArgumentError, "[graph_ir_to_onnx_stub] #{label} #{axis} is out of bounds for rank #{rank}"
      end
      index
    end
    private_class_method :normalize_axis

    def append_aux_int64_initializer!(initializers, used_tensor_names, node_index, label, values)
      normalized = normalize_integer_vector(values, "#{label} for node #{node_index}")
      name = unique_aux_tensor_name(used_tensor_names, node_index, label)
      initializers << onnx_initializer_info(
        "name" => name,
        "shape" => [normalized.length],
        "dtype" => "int64",
        "values" => normalized
      )
      name
    end
    private_class_method :append_aux_int64_initializer!

    def cast_inputs_to_dtype(
      node_index:,
      op_name:,
      inputs:,
      target_dtype:,
      known_shapes:,
      known_dtypes:,
      used_tensor_names:,
      indices: nil
    )
      cast_nodes = []
      casted_inputs = inputs.dup
      cast_to = ONNX_DTYPE_MAP.fetch(target_dtype)
      index_filter = if indices.nil?
        nil
      else
        indices.each_with_object({}) { |value, out| out[value] = true }
      end

      inputs.each_with_index do |input_name, index|
        next if !index_filter.nil? && !index_filter.key?(index)

        input_dtype = canonical_dtype(known_dtypes[input_name])
        next if input_dtype.nil? || input_dtype == target_dtype

        cast_output = unique_aux_tensor_name(used_tensor_names, node_index, "#{op_name.downcase}_input#{index}_cast")
        cast_nodes << build_onnx_node_spec(
          "node_#{node_index}_#{op_name}CastInput#{index}",
          "Cast",
          [input_name],
          [cast_output],
          { "to" => cast_to }
        )
        input_shape = known_shapes[input_name]
        known_shapes[cast_output] = input_shape.dup unless input_shape.nil?
        known_dtypes[cast_output] = target_dtype
        casted_inputs[index] = cast_output
      end

      [casted_inputs, cast_nodes]
    end
    private_class_method :cast_inputs_to_dtype

    def promote_binary_dtype(lhs_dtype, rhs_dtype)
      lhs = canonical_dtype(lhs_dtype)
      rhs = canonical_dtype(rhs_dtype)
      return rhs if lhs.nil?
      return lhs if rhs.nil?
      return lhs if lhs == rhs

      lhs_rank = DTYPE_PROMOTION_RANK[lhs]
      rhs_rank = DTYPE_PROMOTION_RANK[rhs]
      return lhs if lhs_rank.nil? || rhs_rank.nil?

      lhs_rank >= rhs_rank ? lhs : rhs
    end
    private_class_method :promote_binary_dtype

    def canonical_dtype(dtype)
      return nil if dtype.nil?

      dtype == "bool_" ? "bool" : dtype
    end
    private_class_method :canonical_dtype

    def onnx_effective_dtype(dtype)
      canonical = canonical_dtype(dtype)
      return nil if canonical.nil?

      canonical == "bfloat16" ? "float32" : canonical
    end
    private_class_method :onnx_effective_dtype

    def unique_aux_tensor_name(used_tensor_names, node_index, label)
      base = "__mlxir_aux_node#{node_index}_#{label}"
      candidate = base
      suffix = 0
      while used_tensor_names.include?(candidate)
        suffix += 1
        candidate = "#{base}_#{suffix}"
      end
      used_tensor_names.add(candidate)
      candidate
    end
    private_class_method :unique_aux_tensor_name

    def collect_payload_tensor_names(payload)
      names = Set.new
      payload.fetch("inputs").each { |tensor| names.add(tensor.fetch("name")) }
      payload.fetch("constants").each { |tensor| names.add(tensor.fetch("name")) }
      payload.fetch("outputs").each { |tensor| names.add(tensor.fetch("name")) }
      payload.fetch("nodes").each do |node|
        node.fetch("inputs").each { |name| names.add(name) }
        node.fetch("outputs").each { |name| names.add(name) }
      end
      names
    end
    private_class_method :collect_payload_tensor_names

    def collect_known_tensor_shapes(payload)
      shapes = {}
      payload.fetch("inputs").each do |tensor|
        shapes[tensor.fetch("name")] = tensor.fetch("shape").dup
      end
      payload.fetch("constants").each do |tensor|
        shapes[tensor.fetch("name")] = tensor.fetch("shape").dup
      end
      payload.fetch("outputs").each do |tensor|
        shapes[tensor.fetch("name")] = tensor.fetch("shape").dup
      end
      shapes
    end
    private_class_method :collect_known_tensor_shapes

    def collect_known_tensor_dtypes(payload)
      dtypes = {}
      payload.fetch("inputs").each do |tensor|
        dtypes[tensor.fetch("name")] = onnx_effective_dtype(tensor.fetch("dtype"))
      end
      payload.fetch("constants").each do |tensor|
        dtypes[tensor.fetch("name")] = onnx_effective_dtype(tensor.fetch("dtype"))
      end
      payload.fetch("outputs").each do |tensor|
        dtypes[tensor.fetch("name")] = onnx_effective_dtype(tensor.fetch("dtype"))
      end
      dtypes
    end
    private_class_method :collect_known_tensor_dtypes

    def concatenate_axis_from_arguments(arguments, strict: true)
      if arguments.is_a?(Array) && arguments.length == 1
        begin
          return normalized_integer_scalar(arguments.first, "Concatenate axis")
        rescue TypeError, RangeError
          # Handled below.
        end
      end
      return nil unless strict

      raise NotImplementedError,
            "[graph_ir_to_onnx_stub] unsupported Concatenate arguments #{arguments.inspect}; expected [axis]"
    end
    private_class_method :concatenate_axis_from_arguments

    def gather_axis_from_arguments(arguments, strict: true)
      if arguments.is_a?(Array) && !arguments.empty?
        first = arguments.first
        begin
          return normalized_integer_scalar(first, "Gather axis")
        rescue TypeError, RangeError
          # Try vector-encoded axis below.
        end

        if first.is_a?(Array) && first.length == 1
          begin
            return normalized_integer_scalar(first.first, "Gather axis")
          rescue TypeError, RangeError
            # Handled below.
          end
        end
      end
      return nil unless strict

      raise NotImplementedError,
            "[graph_ir_to_onnx_stub] unsupported Gather arguments #{arguments.inspect}; expected first argument to encode axis"
    end
    private_class_method :gather_axis_from_arguments

    def scatter_axis_attributes_from_arguments(arguments, strict: true)
      unless arguments.is_a?(Array) && arguments.length >= 2
        return nil unless strict

        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported ScatterAxis arguments #{arguments.inspect}; expected [mode, axis]"
      end

      mode = arguments[0]
      axis = arguments[1]
      begin
        mode = normalized_integer_scalar(mode, "ScatterAxis mode")
        axis = normalized_integer_scalar(axis, "ScatterAxis axis")
      rescue TypeError, RangeError
        return nil unless strict

        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported ScatterAxis arguments #{arguments.inspect}; mode/axis must be Integer"
      end
      if mode != 1
        return nil unless strict

        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported ScatterAxis mode #{mode.inspect}; only update mode (1) is supported"
      end

      { "axis" => axis }
    end
    private_class_method :scatter_axis_attributes_from_arguments

    def normalize_positive_integer(value, label)
      parsed = Integer(value)
      raise ArgumentError, "#{label} must be a positive Integer" if parsed <= 0

      parsed
    rescue ArgumentError, TypeError
      raise ArgumentError, "#{label} must be a positive Integer"
    end
    private_class_method :normalize_positive_integer
  end
end
