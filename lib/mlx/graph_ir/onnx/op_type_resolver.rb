# frozen_string_literal: true

module MLX
  module GraphIR
    module_function

    def onnx_op_type_for_node(node, strict: true, known_shapes: nil)
      op = node.fetch("op")
      return convolution_onnx_op_type(node.fetch("arguments", []), strict: strict) if op == "Convolution"
      return reduce_onnx_op_type(node.fetch("arguments", []), strict: strict) if op == "Reduce"
      return argreduce_onnx_op_type(node.fetch("arguments", []), strict: strict) if op == "ArgReduce"
      if op == "Flatten"
        return flatten_onnx_op_type(node.fetch("arguments", []), strict: strict, node: node, known_shapes: known_shapes)
      end
      if op == "Concatenate"
        return nil if concatenate_axis_from_arguments(node.fetch("arguments", []), strict: strict).nil?

        return ONNX_OP_MAP.fetch(op)
      end

      mapped = ONNX_OP_MAP[op]
      return mapped unless strict && mapped.nil?

      raise NotImplementedError, "[graph_ir_to_onnx_stub] unsupported op #{op.inspect}"
    end
    private_class_method :onnx_op_type_for_node

    def flatten_onnx_op_type(arguments, strict: true, node: nil, known_shapes: nil)
      unless arguments.is_a?(Array) && arguments.length == 2 && arguments.all? { |value| value.is_a?(Integer) }
        return nil unless strict

        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Flatten arguments #{arguments.inspect}; expected [start_axis, end_axis]"
      end

      if known_shapes && node
        input_name = node.fetch("inputs").first
        input_shape = known_shapes[input_name]
        unless input_shape
          return nil unless strict

          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Flatten for tensor #{input_name.inspect} without known static shape"
        end
        flatten_shape_from_arguments(arguments, input_shape)
      end

      ONNX_OP_MAP.fetch("Flatten")
    end
    private_class_method :flatten_onnx_op_type

    def convolution_onnx_op_type(arguments, strict: true)
      parsed = convolution_attributes_from_arguments(arguments, strict: strict)
      return nil if parsed.nil?

      if parsed.fetch("flip")
        return ONNX_OP_MAP.fetch("ConvolutionTranspose")
      end
      if parsed.fetch("input_dilation").any? { |value| value != 1 }
        return nil unless strict

        raise NotImplementedError,
              "[graph_ir_to_onnx_stub] unsupported Convolution input_dilation #{parsed.fetch("input_dilation").inspect}; " \
              "only all-ones input_dilation is supported for flip=false"
      end

      ONNX_OP_MAP.fetch("Convolution")
    end
    private_class_method :convolution_onnx_op_type

    def reduce_onnx_op_type(arguments, strict: true)
      reduce_code = arguments.is_a?(Array) ? arguments.first : nil
      mapped = REDUCE_CODE_TO_ONNX_OP[reduce_code]
      return mapped unless strict && mapped.nil?

      raise NotImplementedError, "[graph_ir_to_onnx_stub] unsupported Reduce code #{reduce_code.inspect}"
    end
    private_class_method :reduce_onnx_op_type

    def argreduce_onnx_op_type(arguments, strict: true)
      reduce_code = arguments.is_a?(Array) ? arguments.first : nil
      mapped = ARG_REDUCE_CODE_TO_ONNX_OP[reduce_code]
      return mapped unless strict && mapped.nil?

      raise NotImplementedError, "[graph_ir_to_onnx_stub] unsupported ArgReduce code #{reduce_code.inspect}"
    end
    private_class_method :argreduce_onnx_op_type
  end
end
