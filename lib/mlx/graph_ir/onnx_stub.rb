# frozen_string_literal: true

require_relative "onnx/lowerer"
require_relative "onnx/op_type_resolver"

module MLX
  module GraphIR
    module_function

    def to_onnx_stub(payload_or_source, opset: 18, model_name: "mlx_graph")
      payload = validate!(payload_or_source)
      opset_version = normalize_positive_integer(opset, "opset")
      graph_name = model_name.to_s
      raise ArgumentError, "model_name must not be empty" if graph_name.empty?

      initializers = payload.fetch("constants").map { |tensor| onnx_initializer_info(tensor) }
      used_tensor_names = collect_payload_tensor_names(payload)
      known_shapes = collect_known_tensor_shapes(payload)
      known_dtypes = collect_known_tensor_dtypes(payload)
      nodes = payload.fetch("nodes").each_with_index.flat_map do |node, index|
        lowered = lower_onnx_node(node, index, initializers, used_tensor_names, known_shapes, known_dtypes)
        lowered.is_a?(Array) ? lowered : [lowered]
      end

      graph = {
        "name" => graph_name,
        "inputs" => payload.fetch("inputs").map { |tensor| onnx_value_info(tensor) },
        "outputs" => payload.fetch("outputs").map { |tensor| onnx_value_info(tensor) },
        "initializers" => initializers,
        "nodes" => nodes
      }

      {
        "format" => "onnx_stub_v1",
        "ir_version" => IR_VERSION,
        "opset" => opset_version,
        "producer_name" => "mlx-ruby",
        "graph" => graph
      }
    end

    def webgpu_compatibility_report(payload_or_source)
      payload = validate!(payload_or_source)
      probe_initializers = []
      probe_used_tensor_names = collect_payload_tensor_names(payload)
      probe_known_shapes = collect_known_tensor_shapes(payload)
      probe_known_dtypes = collect_known_tensor_dtypes(payload)
      node_support = payload.fetch("nodes").each_with_index.map do |node, index|
        op = node.fetch("op")
        mapped = begin
          mapped_type = onnx_op_type_for_node(node, strict: false, known_shapes: probe_known_shapes)
          raise NotImplementedError if mapped_type.nil?

          lower_onnx_node(node, index, probe_initializers, probe_used_tensor_names, probe_known_shapes, probe_known_dtypes)
          mapped_type
        rescue NotImplementedError, ArgumentError, TypeError
          nil
        end
        {
          "index" => index,
          "op" => op,
          "supported" => !mapped.nil?,
          "onnx_op_type" => mapped
        }
      end

      unsupported = node_support.reject { |entry| entry.fetch("supported") }
      {
        "format" => "webgpu_compat_report_v1",
        "ir_version" => payload.fetch("ir_version"),
        "total_nodes" => node_support.length,
        "supported_nodes" => node_support.length - unsupported.length,
        "unsupported_nodes" => unsupported.length,
        "unsupported_ops" => unsupported.map { |entry| entry.fetch("op") }.uniq.sort,
        "ready_for_stub_conversion" => unsupported.empty?,
        "nodes" => node_support
      }
    end

    def onnx_value_info(tensor)
      dtype = onnx_effective_dtype(tensor.fetch("dtype"))
      mapped = ONNX_DTYPE_MAP.fetch(dtype)
      {
        "name" => tensor.fetch("name"),
        "shape" => tensor.fetch("shape"),
        "dtype" => dtype,
        "onnx_elem_type" => mapped
      }
    end
    private_class_method :onnx_value_info

    def onnx_initializer_info(tensor)
      info = onnx_value_info(tensor)
      values = tensor.fetch("values")
      if info["dtype"] == "int64"
        values = normalize_initializer_int64_values(values, "initializer #{info['name']}")
      end
      info["values"] = values
      info
    end
    private_class_method :onnx_initializer_info

    def normalize_initializer_int64_values(value, label)
      case value
      when Array
        value.map { |item| normalize_initializer_int64_values(item, label) }
      else
        normalized_integer_scalar(value, label)
      end
    end
    private_class_method :normalize_initializer_int64_values

    def onnx_node_attributes(node)
      op = node.fetch("op")
      arguments = node.fetch("arguments", [])
      case op
      when "Transpose"
        perm = transpose_perm_from_arguments(arguments)
        return {} if perm.nil? || perm.empty?

        { "perm" => perm }
      when "Concatenate"
        { "axis" => concatenate_axis_from_arguments(arguments) }
      when "Gather"
        { "axis" => gather_axis_from_arguments(arguments) }
      when "GatherAxis"
        { "axis" => gather_axis_from_arguments(arguments) }
      when "ScatterAxis"
        scatter_axis_attributes_from_arguments(arguments)
      else
        {}
      end
    end
    private_class_method :onnx_node_attributes

    def build_onnx_node_spec(name, op_type, inputs, outputs, attributes)
      {
        "name" => name,
        "op_type" => op_type,
        "inputs" => inputs,
        "outputs" => outputs,
        "attributes" => attributes
      }
    end
    private_class_method :build_onnx_node_spec

    def transpose_perm_from_arguments(arguments)
      candidate = arguments.find do |value|
        next false unless value.is_a?(Array)

        begin
          normalize_integer_vector(value, "Transpose permutation")
          true
        rescue TypeError, RangeError
          false
        end
      end
      return nil if candidate.nil?

      normalize_integer_vector(candidate, "Transpose permutation")
    end
    private_class_method :transpose_perm_from_arguments



    def convolution_attributes_from_arguments(arguments, strict: true)
      unless arguments.is_a?(Array) && arguments.length >= 7
        return nil unless strict

        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Convolution arguments #{arguments.inspect}; " \
              "expected [strides, padding_low, padding_high, kernel_dilation, input_dilation, groups, flip]"
      end

      strides = normalize_integer_vector(arguments[0], "Convolution strides")
      padding_low = normalize_integer_vector(arguments[1], "Convolution padding_low")
      padding_high = normalize_integer_vector(arguments[2], "Convolution padding_high")
      kernel_dilation = normalize_integer_vector(arguments[3], "Convolution kernel_dilation")
      input_dilation = normalize_integer_vector(arguments[4], "Convolution input_dilation")
      groups = arguments[5]
      flip = arguments[6]

      lengths = [
        padding_low.length,
        padding_high.length,
        kernel_dilation.length,
        input_dilation.length
      ]
      unless lengths.all? { |length| length == strides.length }
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] Convolution argument lengths must match spatial rank #{strides.length}: " \
              "padding_low=#{padding_low.length}, padding_high=#{padding_high.length}, " \
              "kernel_dilation=#{kernel_dilation.length}, input_dilation=#{input_dilation.length}"
      end
      if strides.any? { |value| value <= 0 }
        raise ArgumentError, "[graph_ir_to_onnx_stub] Convolution strides must be positive"
      end
      if padding_low.any?(&:negative?) || padding_high.any?(&:negative?)
        raise NotImplementedError, "[graph_ir_to_onnx_stub] unsupported Convolution with negative padding"
      end
      if kernel_dilation.any? { |value| value <= 0 }
        raise ArgumentError, "[graph_ir_to_onnx_stub] Convolution kernel_dilation must be positive"
      end
      if input_dilation.any? { |value| value <= 0 }
        raise ArgumentError, "[graph_ir_to_onnx_stub] Convolution input_dilation must be positive"
      end
      unless groups.is_a?(Integer) && groups > 0
        raise TypeError, "[graph_ir_to_onnx_stub] Convolution groups must be a positive Integer"
      end
      unless flip == true || flip == false
        raise TypeError, "[graph_ir_to_onnx_stub] Convolution flip must be boolean"
      end

      {
        "strides" => strides,
        "padding_low" => padding_low,
        "padding_high" => padding_high,
        "pads" => [*padding_low, *padding_high],
        "kernel_dilation" => kernel_dilation,
        "input_dilation" => input_dilation,
        "groups" => groups,
        "flip" => flip,
        "spatial_rank" => strides.length
      }
    end
    private_class_method :convolution_attributes_from_arguments

    def convtranspose_attributes_from_convolution(convolution, weight_shape)
      weight = normalize_integer_vector(weight_shape, "ConvolutionTranspose weight shape")
      spatial_rank = convolution.fetch("spatial_rank")
      expected_rank = spatial_rank + 2
      unless weight.length == expected_rank
        raise ArgumentError,
              "[graph_ir_to_onnx_stub] ConvolutionTranspose weight rank mismatch: expected #{expected_rank}, got #{weight.length}"
      end

      kernel_shape = weight[1..-2]
      kernel_dilation = convolution.fetch("kernel_dilation")
      padding_low = convolution.fetch("padding_low")
      padding_high = convolution.fetch("padding_high")
      strides = convolution.fetch("input_dilation")

      base_padding = kernel_shape.each_with_index.map do |kernel_dim, axis|
        (kernel_dilation[axis] * (kernel_dim - 1))
      end
      pads_begin = base_padding.each_with_index.map do |base, axis|
        base - padding_low[axis]
      end
      output_padding = padding_high.each_with_index.map do |high, axis|
        high - padding_low[axis]
      end
      pads_end = base_padding.each_with_index.map do |base, axis|
        base - padding_high[axis] + output_padding[axis]
      end

      if pads_begin.any?(&:negative?) || pads_end.any?(&:negative?)
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported ConvolutionTranspose derived negative padding from arguments"
      end
      if output_padding.any?(&:negative?)
        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported ConvolutionTranspose with negative output_padding"
      end
      output_padding.each_with_index do |value, axis|
        stride = strides[axis]
        next if value < stride

        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported ConvolutionTranspose output_padding #{output_padding.inspect}; " \
              "each value must be < corresponding stride #{strides.inspect}"
      end

      {
        "strides" => strides,
        "dilations" => kernel_dilation,
        "pads_begin" => pads_begin,
        "pads_end" => pads_end,
        "pads" => [*pads_begin, *pads_end],
        "output_padding" => output_padding
      }
    end
    private_class_method :convtranspose_attributes_from_convolution
  end
end
