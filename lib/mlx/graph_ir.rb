# frozen_string_literal: true

require "json"
require "set"

module MLX
  module GraphIR
    module_function

    IR_VERSION = 1

    REQUIRED_TOP_LEVEL_KEYS = %w[
      ir_version
      shapeless
      inputs
      keyword_inputs
      outputs
      constants
      nodes
    ].freeze

    SUPPORTED_DTYPES = %w[
      bool
      bool_
      uint8
      uint16
      uint32
      uint64
      int8
      int16
      int32
      int64
      float16
      float32
      float64
      bfloat16
      complex64
    ].freeze

    ONNX_OP_MAP = {
      "Add" => "Add",
      "AddMM" => "Gemm",
      "Subtract" => "Sub",
      "Multiply" => "Mul",
      "Square" => "Mul",
      "Divide" => "Div",
      "Exp" => "Exp",
      "Log" => "Log",
      "Sqrt" => "Sqrt",
      "Abs" => "Abs",
      "Negative" => "Neg",
      "Relu" => "Relu",
      "Sigmoid" => "Sigmoid",
      "Tanh" => "Tanh",
      "Softmax" => "Softmax",
      "Greater" => "Greater",
      "Select" => "Where",
      "Full" => "Identity",
      "Matmul" => "MatMul",
      "Reshape" => "Reshape",
      "Flatten" => "Reshape",
      "Unflatten" => "Reshape",
      "Transpose" => "Transpose",
      "Squeeze" => "Squeeze",
      "ExpandDims" => "Unsqueeze",
      "Broadcast" => "Expand",
      "Arange" => "Constant",
      "AsStrided" => "Gather",
      "Concatenate" => "Concat",
      "Convolution" => "Conv",
      "ConvolutionTranspose" => "ConvTranspose",
      "Gather" => "Gather",
      "Slice" => "Slice",
      "Split" => "Split",
      "ScatterAxis" => "ScatterElements",
      "Maximum" => "Max",
      "Minimum" => "Min",
      "Power" => "Pow"
    }.freeze

    REDUCE_CODE_TO_ONNX_OP = {
      0 => "ReduceMin",
      1 => "ReduceMax",
      2 => "ReduceSum",
      3 => "ReduceProd",
      4 => "ReduceMin",
      5 => "ReduceMax"
    }.freeze

    ARG_REDUCE_CODE_TO_ONNX_OP = {
      0 => "ArgMin",
      1 => "ArgMax"
    }.freeze

    ONNX_DTYPE_MAP = {
      "bool" => "BOOL",
      "bool_" => "BOOL",
      "uint8" => "UINT8",
      "uint16" => "UINT16",
      "uint32" => "UINT32",
      "uint64" => "UINT64",
      "int8" => "INT8",
      "int16" => "INT16",
      "int32" => "INT32",
      "int64" => "INT64",
      "float16" => "FLOAT16",
      "float32" => "FLOAT",
      "float64" => "DOUBLE",
      "bfloat16" => "BFLOAT16",
      "complex64" => "COMPLEX64"
    }.freeze

    def load_payload(payload_or_source)
      payload = case payload_or_source
      when Hash
        deep_copy_hash(payload_or_source)
      when String
        if File.file?(payload_or_source)
          parse_json_payload(File.binread(payload_or_source), "graph ir file")
        else
          parse_json_payload(payload_or_source, "graph ir string")
        end
      else
        if payload_or_source.respond_to?(:read)
          parse_json_payload(payload_or_source.read.to_s, "graph ir IO")
        else
          raise TypeError, "graph ir source must be a Hash, JSON String, file path, or IO-like object"
        end
      end
      normalize_hash_keys(payload)
    end

    def validate!(payload_or_source)
      payload = load_payload(payload_or_source)
      validate_top_level!(payload)
      validate_graph_topology!(payload)
      payload
    end

    def to_onnx_stub(payload_or_source, opset: 18, model_name: "mlx_graph")
      payload = validate!(payload_or_source)
      opset_version = normalize_positive_integer(opset, "opset")
      graph_name = model_name.to_s
      raise ArgumentError, "model_name must not be empty" if graph_name.empty?

      initializers = payload.fetch("constants").map { |tensor| onnx_initializer_info(tensor) }
      used_tensor_names = collect_payload_tensor_names(payload)
      known_shapes = collect_known_tensor_shapes(payload)
      nodes = payload.fetch("nodes").each_with_index.flat_map do |node, index|
        lowered = lower_onnx_node(node, index, initializers, used_tensor_names, known_shapes)
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
      node_support = payload.fetch("nodes").each_with_index.map do |node, index|
        op = node.fetch("op")
        mapped = begin
          mapped_type = onnx_op_type_for_node(node, strict: false, known_shapes: probe_known_shapes)
          raise NotImplementedError if mapped_type.nil?

          lower_onnx_node(node, index, probe_initializers, probe_used_tensor_names, probe_known_shapes)
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

    def dump_json(hash, pretty: true)
      pretty ? JSON.pretty_generate(hash) : JSON.generate(hash)
    end

    def onnx_json_compatible_value(value)
      case value
      when Hash
        value.each_with_object({}) do |(key, item), out|
          out[key] = onnx_json_compatible_value(item)
        end
      when Array
        value.map { |item| onnx_json_compatible_value(item) }
      when ::Complex
        {
          "__mlx_complex__" => [
            value.real.to_f,
            value.imag.to_f
          ]
        }
      when String
        parsed = parse_ruby_complex_literal(value)
        parsed ? { "__mlx_complex__" => parsed } : value
      else
        value
      end
    end

    def normalize_hash_keys(value)
      case value
      when Hash
        value.each_with_object({}) do |(key, item), out|
          out[key.to_s] = normalize_hash_keys(item)
        end
      when Array
        value.map { |item| normalize_hash_keys(item) }
      else
        value
      end
    end
    private_class_method :normalize_hash_keys

    def deep_copy_hash(value)
      case value
      when Hash
        value.each_with_object({}) do |(key, item), out|
          out[key] = deep_copy_hash(item)
        end
      when Array
        value.map { |item| deep_copy_hash(item) }
      else
        value
      end
    end
    private_class_method :deep_copy_hash

    def parse_ruby_complex_literal(value)
      return nil unless value.include?("i")

      complex = begin
        Complex(value)
      rescue ArgumentError, TypeError
        nil
      end
      return nil if complex.nil?

      [complex.real.to_f, complex.imag.to_f]
    end
    private_class_method :parse_ruby_complex_literal

    def parse_json_payload(raw, label)
      JSON.parse(raw)
    rescue JSON::ParserError => e
      raise ArgumentError, "failed to parse #{label}: #{e.message}"
    end
    private_class_method :parse_json_payload

    def validate_top_level!(payload)
      unless payload.is_a?(Hash)
        raise TypeError, "graph ir must be a Hash at top-level"
      end

      missing = REQUIRED_TOP_LEVEL_KEYS - payload.keys
      unless missing.empty?
        raise ArgumentError, "graph ir missing required keys: #{missing.join(", ")}"
      end

      unknown = payload.keys - REQUIRED_TOP_LEVEL_KEYS
      unless unknown.empty?
        raise ArgumentError, "graph ir contains unknown keys: #{unknown.join(", ")}"
      end

      unless payload.fetch("ir_version") == IR_VERSION
        raise ArgumentError, "graph ir ir_version must be #{IR_VERSION}"
      end

      shapeless = payload.fetch("shapeless")
      unless shapeless == true || shapeless == false
        raise TypeError, "graph ir shapeless must be true or false"
      end

      validate_tensor_array!(payload.fetch("inputs"), "inputs")
      validate_keyword_inputs!(payload.fetch("keyword_inputs"))
      validate_tensor_array!(payload.fetch("outputs"), "outputs")
      validate_constants_array!(payload.fetch("constants"))
      validate_nodes_array!(payload.fetch("nodes"))
    end
    private_class_method :validate_top_level!

    def validate_tensor_array!(value, label)
      unless value.is_a?(Array)
        raise TypeError, "graph ir #{label} must be an Array"
      end
      value.each_with_index do |tensor, index|
        validate_tensor_info!(tensor, "#{label}[#{index}]")
      end
    end
    private_class_method :validate_tensor_array!

    def validate_tensor_info!(tensor, path)
      unless tensor.is_a?(Hash)
        raise TypeError, "#{path} must be a Hash"
      end

      required = %w[name shape dtype]
      missing = required - tensor.keys
      unless missing.empty?
        raise ArgumentError, "#{path} missing required keys: #{missing.join(", ")}"
      end

      unknown = tensor.keys - required
      unless unknown.empty?
        raise ArgumentError, "#{path} contains unknown keys: #{unknown.join(", ")}"
      end

      name = tensor.fetch("name")
      unless name.is_a?(String) && !name.empty?
        raise TypeError, "#{path}.name must be a non-empty String"
      end

      shape = tensor.fetch("shape")
      unless shape.is_a?(Array)
        raise TypeError, "#{path}.shape must be an Array of non-negative Integer"
      end
      shape.each_with_index do |dim, dim_index|
        unless dim.is_a?(Integer) && dim >= 0
          raise TypeError, "#{path}.shape[#{dim_index}] must be a non-negative Integer"
        end
      end

      dtype = tensor.fetch("dtype")
      unless dtype.is_a?(String) && !dtype.empty?
        raise TypeError, "#{path}.dtype must be a non-empty String"
      end
      unless SUPPORTED_DTYPES.include?(dtype)
        raise ArgumentError, "#{path}.dtype #{dtype.inspect} is not supported"
      end
    end
    private_class_method :validate_tensor_info!

    def validate_constants_array!(value)
      unless value.is_a?(Array)
        raise TypeError, "graph ir constants must be an Array"
      end
      value.each_with_index do |constant, index|
        validate_constant_info!(constant, "constants[#{index}]")
      end
    end
    private_class_method :validate_constants_array!

    def validate_constant_info!(tensor, path)
      unless tensor.is_a?(Hash)
        raise TypeError, "#{path} must be a Hash"
      end

      required = %w[name shape dtype values]
      missing = required - tensor.keys
      unless missing.empty?
        raise ArgumentError, "#{path} missing required keys: #{missing.join(", ")}"
      end

      unknown = tensor.keys - required
      unless unknown.empty?
        raise ArgumentError, "#{path} contains unknown keys: #{unknown.join(", ")}"
      end

      validate_tensor_info!(tensor.slice("name", "shape", "dtype"), path)
      validate_constant_values_shape!(
        tensor.fetch("values"),
        tensor.fetch("shape"),
        path,
        tensor.fetch("dtype")
      )
    end
    private_class_method :validate_constant_info!

    def validate_constant_values_shape!(values, shape, path, dtype)
      expected = shape.empty? ? 1 : shape.reduce(1, :*)
      actual = count_constant_values(values, "#{path}.values", dtype)
      unless actual == expected
        raise ArgumentError,
              "#{path}.values has #{actual} value(s), expected #{expected} based on shape #{shape.inspect}"
      end
    end
    private_class_method :validate_constant_values_shape!

    def count_constant_values(value, path, dtype)
      if value.is_a?(Array)
        return value.each_with_index.sum do |item, index|
          count_constant_values(item, "#{path}[#{index}]", dtype)
        end
      end

      if value.is_a?(Numeric) || value == true || value == false
        return 1
      end

      if dtype == "complex64" && complex_json_compatible_leaf?(value)
        return 1
      end

      raise TypeError, "#{path} leaf values must be Numeric or boolean"
    end
    private_class_method :count_constant_values

    def complex_json_compatible_leaf?(value)
      return !parse_ruby_complex_literal(value).nil? if value.is_a?(String)
      return false unless value.is_a?(Hash)

      return false unless value.keys == ["__mlx_complex__"]

      pair = value["__mlx_complex__"]
      return false unless pair.is_a?(Array) && pair.length == 2

      pair.all? { |item| float_like?(item) }
    end
    private_class_method :complex_json_compatible_leaf?

    def float_like?(value)
      return true if value.is_a?(Numeric)

      begin
        Float(value)
      rescue ArgumentError, TypeError
        false
      else
        true
      end
    end
    private_class_method :float_like?

    def validate_keyword_inputs!(value)
      unless value.is_a?(Array)
        raise TypeError, "graph ir keyword_inputs must be an Array"
      end
      value.each_with_index do |entry, index|
        path = "keyword_inputs[#{index}]"
        unless entry.is_a?(Hash)
          raise TypeError, "#{path} must be a Hash"
        end
        required = %w[name tensor]
        missing = required - entry.keys
        unless missing.empty?
          raise ArgumentError, "#{path} missing required keys: #{missing.join(", ")}"
        end
        unknown = entry.keys - required
        unless unknown.empty?
          raise ArgumentError, "#{path} contains unknown keys: #{unknown.join(", ")}"
        end
        unless entry.fetch("name").is_a?(String) && !entry.fetch("name").empty?
          raise TypeError, "#{path}.name must be a non-empty String"
        end
        unless entry.fetch("tensor").is_a?(String) && !entry.fetch("tensor").empty?
          raise TypeError, "#{path}.tensor must be a non-empty String"
        end
      end
    end
    private_class_method :validate_keyword_inputs!

    def validate_nodes_array!(value)
      unless value.is_a?(Array)
        raise TypeError, "graph ir nodes must be an Array"
      end
      value.each_with_index do |node, index|
        path = "nodes[#{index}]"
        unless node.is_a?(Hash)
          raise TypeError, "#{path} must be a Hash"
        end
        required = %w[op inputs outputs]
        missing = required - node.keys
        unless missing.empty?
          raise ArgumentError, "#{path} missing required keys: #{missing.join(", ")}"
        end
        optional = %w[arguments]
        unknown = node.keys - required - optional
        unless unknown.empty?
          raise ArgumentError, "#{path} contains unknown keys: #{unknown.join(", ")}"
        end
        op = node.fetch("op")
        unless op.is_a?(String) && !op.empty?
          raise TypeError, "#{path}.op must be a non-empty String"
        end
        %w[inputs outputs].each do |field|
          refs = node.fetch(field)
          unless refs.is_a?(Array)
            raise TypeError, "#{path}.#{field} must be an Array of non-empty String"
          end
          if field == "outputs" && refs.empty?
            raise ArgumentError, "#{path}.outputs must contain at least one tensor name"
          end
          refs.each_with_index do |name, name_index|
            unless name.is_a?(String) && !name.empty?
              raise TypeError, "#{path}.#{field}[#{name_index}] must be a non-empty String"
            end
          end
        end

        if node.key?("arguments")
          validate_node_arguments!(node.fetch("arguments"), "#{path}.arguments")
        end
      end
    end
    private_class_method :validate_nodes_array!

    def validate_node_arguments!(arguments, path)
      unless arguments.is_a?(Array)
        raise TypeError, "#{path} must be an Array"
      end

      arguments.each_with_index do |value, index|
        validate_state_value!(value, "#{path}[#{index}]")
      end
    end
    private_class_method :validate_node_arguments!

    def validate_state_value!(value, path)
      if value.nil? || value == true || value == false || value.is_a?(Numeric) || value.is_a?(String)
        return
      end
      if value.is_a?(Array)
        value.each_with_index do |item, index|
          validate_state_value!(item, "#{path}[#{index}]")
        end
        return
      end
      raise TypeError, "#{path} must be nil, boolean, Numeric, String, or nested Arrays of those values"
    end
    private_class_method :validate_state_value!

    def validate_graph_topology!(payload)
      input_names = collect_unique_tensor_names(payload.fetch("inputs"), "inputs")
      constant_names = collect_unique_tensor_names(payload.fetch("constants"), "constants")
      keyword_inputs = payload.fetch("keyword_inputs")
      keyword_inputs.each_with_index do |entry, index|
        tensor_name = entry.fetch("tensor")
        unless input_names.include?(tensor_name)
          raise ArgumentError, "keyword_inputs[#{index}].tensor references unknown input #{tensor_name.inspect}"
        end
      end

      available = Set.new
      input_names.each { |name| available.add(name) }
      constant_names.each do |name|
        if available.include?(name)
          raise ArgumentError, "constants tensor #{name.inspect} collides with existing tensor name"
        end
        available.add(name)
      end

      payload.fetch("nodes").each_with_index do |node, node_index|
        node.fetch("inputs").each_with_index do |input_name, input_index|
          unless available.include?(input_name)
            raise ArgumentError,
                  "nodes[#{node_index}].inputs[#{input_index}] references unknown tensor #{input_name.inspect}"
          end
        end
        local_outputs = Set.new
        node.fetch("outputs").each_with_index do |output_name, output_index|
          if local_outputs.include?(output_name)
            raise ArgumentError,
                  "nodes[#{node_index}].outputs[#{output_index}] duplicates output #{output_name.inspect} in the same node"
          end
          local_outputs.add(output_name)

          if available.include?(output_name)
            raise ArgumentError,
                  "nodes[#{node_index}].outputs[#{output_index}] redefines existing tensor #{output_name.inspect}"
          end
        end
        local_outputs.each { |name| available.add(name) }
      end

      collect_unique_tensor_names(payload.fetch("outputs"), "outputs").each_with_index do |name, output_index|
        unless available.include?(name)
          raise ArgumentError, "outputs[#{output_index}] references unknown tensor #{name.inspect}"
        end
      end
    end
    private_class_method :validate_graph_topology!

    def collect_unique_tensor_names(infos, label)
      names = Set.new
      ordered = []
      infos.each_with_index do |info, index|
        name = info.fetch("name")
        if names.include?(name)
          raise ArgumentError, "#{label}[#{index}].name duplicates tensor #{name.inspect}"
        end
        names.add(name)
        ordered << name
      end
      ordered
    end
    private_class_method :collect_unique_tensor_names

    def onnx_value_info(tensor)
      dtype = tensor.fetch("dtype")
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
      info["values"] = tensor.fetch("values")
      info
    end
    private_class_method :onnx_initializer_info

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
        value.is_a?(Array) && value.all? { |item| item.is_a?(Integer) }
      end
      return nil if candidate.nil?

      candidate
    end
    private_class_method :transpose_perm_from_arguments

    def lower_onnx_node(node, node_index, initializers, used_tensor_names, known_shapes)
      op = node.fetch("op")
      op_type = onnx_op_type_for_node(node, known_shapes: known_shapes)
      unless op_type
        raise NotImplementedError, "[graph_ir_to_onnx_stub] unsupported op #{op.inspect}"
      end

      inputs = node.fetch("inputs").dup
      attributes = onnx_node_attributes(node)
      arguments = node.fetch("arguments", [])
      outputs = node.fetch("outputs")
      inferred_output_shape = nil

      case op
      when "Arange"
        start, stop, step = arange_arguments(arguments)
        values = arange_values(start, stop, step)
        output_name = outputs.fetch(0)
        initializers << onnx_initializer_info(
          "name" => output_name,
          "shape" => [values.length],
          "dtype" => "int64",
          "values" => values
        )
        known_shapes[output_name] = [values.length]
        return []
      when "Transpose"
        transpose_input = inputs.fetch(0)
        input_shape = known_shapes[transpose_input]
        unless input_shape.nil?
          input_dims = normalize_integer_vector(input_shape, "Transpose input shape")
          perm = attributes["perm"]
          perm = (0...input_dims.length).to_a.reverse if perm.nil? || perm.empty?
          inferred_output_shape = permute_shape(input_dims, perm, "Transpose permutation")
        end
      when "Convolution"
        convolution = convolution_attributes_from_arguments(arguments)
        if convolution.fetch("flip")
          spatial_rank = convolution.fetch("spatial_rank")
          input_perm, output_perm = convolution_data_permutations(spatial_rank)
          weight_perm = convolution_transpose_weight_permutation(spatial_rank)

          input_shape = known_shapes[inputs[0]]
          weight_shape = known_shapes[inputs[1]]
          unless weight_shape
            raise NotImplementedError,
                  "[graph_ir_to_onnx_stub] unsupported Convolution flip=true without known static weight shape"
          end

          transposed_input = unique_aux_tensor_name(used_tensor_names, node_index, "conv_transpose_input_ncx")
          transposed_weight = unique_aux_tensor_name(used_tensor_names, node_index, "conv_transpose_weight_icx")
          conv_output = unique_aux_tensor_name(used_tensor_names, node_index, "conv_transpose_output_ncx")

          unless input_shape.nil?
            known_shapes[transposed_input] = permute_shape(
              input_shape,
              input_perm,
              "ConvolutionTranspose input permutation"
            )
          end
          known_shapes[transposed_weight] = permute_shape(
            weight_shape,
            weight_perm,
            "ConvolutionTranspose weight permutation"
          )

          conv_transpose = convtranspose_attributes_from_convolution(convolution, weight_shape)
          inferred_output_shape = infer_convolution_transpose_output_shape(
            input_shape,
            weight_shape,
            conv_transpose.fetch("strides"),
            conv_transpose.fetch("pads_begin"),
            conv_transpose.fetch("pads_end"),
            conv_transpose.fetch("dilations"),
            conv_transpose.fetch("output_padding"),
            convolution.fetch("groups")
          )
          unless inferred_output_shape.nil?
            known_shapes[conv_output] = permute_shape(
              inferred_output_shape,
              input_perm,
              "ConvolutionTranspose output permutation"
            )
            outputs.each do |name|
              known_shapes[name] = inferred_output_shape.dup
            end
          end

          conv_transpose_attributes = {
            "strides" => conv_transpose.fetch("strides"),
            "pads" => conv_transpose.fetch("pads"),
            "dilations" => conv_transpose.fetch("dilations"),
            "group" => convolution.fetch("groups"),
            "output_padding" => conv_transpose.fetch("output_padding")
          }

          return [
            build_onnx_node_spec(
              "node_#{node_index}_InputTranspose",
              "Transpose",
              [inputs[0]],
              [transposed_input],
              { "perm" => input_perm }
            ),
            build_onnx_node_spec(
              "node_#{node_index}_WeightTranspose",
              "Transpose",
              [inputs[1]],
              [transposed_weight],
              { "perm" => weight_perm }
            ),
            build_onnx_node_spec(
              "node_#{node_index}_ConvTranspose",
              "ConvTranspose",
              [transposed_input, transposed_weight],
              [conv_output],
              conv_transpose_attributes
            ),
            build_onnx_node_spec(
              "node_#{node_index}_OutputTranspose",
              "Transpose",
              [conv_output],
              outputs,
              { "perm" => output_perm }
            )
          ]
        end
        if convolution.fetch("input_dilation").any? { |value| value != 1 }
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Convolution input_dilation #{convolution.fetch("input_dilation").inspect}; " \
                "only all-ones input_dilation is supported for flip=false"
        end

        spatial_rank = convolution.fetch("spatial_rank")
        input_perm, output_perm = convolution_data_permutations(spatial_rank)
        weight_perm = convolution_weight_permutation(spatial_rank)

        input_shape = known_shapes[inputs[0]]
        weight_shape = known_shapes[inputs[1]]
        input_rank = spatial_rank + 2
        if input_shape && normalize_integer_vector(input_shape, "Convolution input shape").length != input_rank
          raise ArgumentError,
                "[graph_ir_to_onnx_stub] Convolution input rank mismatch: expected #{input_rank}, got #{input_shape.length}"
        end
        if weight_shape && normalize_integer_vector(weight_shape, "Convolution weight shape").length != input_rank
          raise ArgumentError,
                "[graph_ir_to_onnx_stub] Convolution weight rank mismatch: expected #{input_rank}, got #{weight_shape.length}"
        end

        transposed_input = unique_aux_tensor_name(used_tensor_names, node_index, "conv_input_ncx")
        transposed_weight = unique_aux_tensor_name(used_tensor_names, node_index, "conv_weight_ocx")
        conv_output = unique_aux_tensor_name(used_tensor_names, node_index, "conv_output_ncx")

        unless input_shape.nil?
          known_shapes[transposed_input] = permute_shape(
            input_shape,
            input_perm,
            "Convolution input permutation"
          )
        end
        unless weight_shape.nil?
          known_shapes[transposed_weight] = permute_shape(
            weight_shape,
            weight_perm,
            "Convolution weight permutation"
          )
        end

        inferred_output_shape = infer_convolution_output_shape(
          input_shape,
          weight_shape,
          convolution.fetch("strides"),
          convolution.fetch("padding_low"),
          convolution.fetch("padding_high"),
          convolution.fetch("kernel_dilation"),
          convolution.fetch("groups")
        )
        unless inferred_output_shape.nil?
          known_shapes[conv_output] = permute_shape(
            inferred_output_shape,
            input_perm,
            "Convolution output permutation"
          )
          outputs.each do |name|
            known_shapes[name] = inferred_output_shape.dup
          end
        end

        conv_attributes = {
          "strides" => convolution.fetch("strides"),
          "pads" => convolution.fetch("pads"),
          "dilations" => convolution.fetch("kernel_dilation"),
          "group" => convolution.fetch("groups")
        }

        return [
          build_onnx_node_spec(
            "node_#{node_index}_InputTranspose",
            "Transpose",
            [inputs[0]],
            [transposed_input],
            { "perm" => input_perm }
          ),
          build_onnx_node_spec(
            "node_#{node_index}_WeightTranspose",
            "Transpose",
            [inputs[1]],
            [transposed_weight],
            { "perm" => weight_perm }
          ),
          build_onnx_node_spec(
            "node_#{node_index}_Conv",
            "Conv",
            [transposed_input, transposed_weight],
            [conv_output],
            conv_attributes
          ),
          build_onnx_node_spec(
            "node_#{node_index}_OutputTranspose",
            "Transpose",
            [conv_output],
            outputs,
            { "perm" => output_perm }
          )
        ]
      when "Reduce"
        reduce_code = arguments.first
        axes = reduce_axes_from_arguments(arguments)
        axes_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axes", axes)
        inferred_output_shape = infer_reduce_keepdims_shape(known_shapes[inputs.first], axes)

        if reduce_code == 0 || reduce_code == 1
          cast_bool_out = unique_aux_tensor_name(used_tensor_names, node_index, "cast_bool")
          cast_int_out = unique_aux_tensor_name(used_tensor_names, node_index, "cast_int64")
          reduce_out = unique_aux_tensor_name(used_tensor_names, node_index, "reduce")
          reduce_type = REDUCE_CODE_TO_ONNX_OP.fetch(reduce_code)

          outputs.each do |name|
            known_shapes[name] = inferred_output_shape.dup unless inferred_output_shape.nil?
          end

          return [
            build_onnx_node_spec(
              "node_#{node_index}_CastToBool",
              "Cast",
              [inputs.first],
              [cast_bool_out],
              { "to" => "BOOL" }
            ),
            build_onnx_node_spec(
              "node_#{node_index}_CastToInt64",
              "Cast",
              [cast_bool_out],
              [cast_int_out],
              { "to" => "INT64" }
            ),
            build_onnx_node_spec(
              "node_#{node_index}_#{reduce_type}",
              reduce_type,
              [cast_int_out, axes_name],
              [reduce_out],
              { "keepdims" => 1 }
            ),
            build_onnx_node_spec(
              "node_#{node_index}_CastOutBool",
              "Cast",
              [reduce_out],
              outputs,
              { "to" => "BOOL" }
            )
          ]
        end

        inputs << axes_name
        attributes["keepdims"] = 1
      when "Reshape"
        shape = integer_vector_argument(arguments, "Reshape")
        shape_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "shape", shape)
        inputs << shape_name
        inferred_output_shape = shape
      when "Add", "Subtract", "Multiply", "Divide", "Maximum", "Minimum", "Power"
        inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
      when "Exp", "Log", "Abs", "Negative", "Relu", "Sigmoid", "Tanh", "Softmax"
        inferred_output_shape = known_shapes[inputs[0]]
      when "Sqrt"
        inferred_output_shape = known_shapes[inputs[0]]
        if sqrt_is_reciprocal?(arguments)
          sqrt_output = unique_aux_tensor_name(used_tensor_names, node_index, "sqrt")
          known_shapes[sqrt_output] = inferred_output_shape.dup unless inferred_output_shape.nil?
          return [
            build_onnx_node_spec(
              "node_#{node_index}_Sqrt",
              "Sqrt",
              inputs,
              [sqrt_output],
              {}
            ),
            build_onnx_node_spec(
              "node_#{node_index}_Reciprocal",
              "Reciprocal",
              [sqrt_output],
              outputs,
              {}
            )
          ]
        end
      when "Matmul"
        inferred_output_shape = infer_matmul_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
      when "AddMM"
        alpha, beta = addmm_alpha_beta(arguments)
        attributes["alpha"] = alpha unless alpha == 1.0
        attributes["beta"] = beta unless beta == 1.0
        attributes["transA"] = 0
        attributes["transB"] = 0
        inferred_output_shape = infer_matmul_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
      when "Square"
        square_input = inputs.fetch(0)
        inputs = [square_input, square_input]
        inferred_output_shape = known_shapes[square_input]
      when "Gather"
        axis = gather_axis_from_arguments(arguments)
        data_shape = known_shapes[inputs[0]]
        indices_shape = known_shapes[inputs[1]]
        unless data_shape
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Gather for tensor #{inputs[0].inspect} without known static shape"
        end
        unless indices_shape
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Gather for indices #{inputs[1].inspect} without known static shape"
        end
        data_rank = normalize_integer_vector(data_shape, "Gather data shape").length
        axis_index = normalize_axis(axis, data_rank, "Gather axis")
        indices_rank = normalize_integer_vector(indices_shape, "Gather indices shape").length
        gather_output = unique_aux_tensor_name(used_tensor_names, node_index, "gather")
        gather_reordered = unique_aux_tensor_name(used_tensor_names, node_index, "gather_reordered")
        unsqueeze_axis = axis_index + indices_rank
        axes_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axes", [unsqueeze_axis])

        gather_shape = infer_gather_output_shape(data_shape, indices_shape, axis_index)
        perm = gather_reorder_permutation(data_rank, indices_rank, axis_index)
        needs_reorder = !identity_permutation?(perm)
        reordered_shape = gather_shape.nil? ? nil : perm.map { |dim_index| gather_shape[dim_index] }
        unless gather_shape.nil?
          known_shapes[gather_output] = gather_shape
          known_shapes[gather_reordered] = reordered_shape if needs_reorder
          final_shape = reordered_shape.dup
          final_shape.insert(unsqueeze_axis, 1)
          outputs.each { |name| known_shapes[name] = final_shape.dup }
        end

        lowered = [
          build_onnx_node_spec("node_#{node_index}_Gather", "Gather", inputs, [gather_output], attributes)
        ]
        unsqueeze_input = gather_output
        if needs_reorder
          lowered << build_onnx_node_spec(
            "node_#{node_index}_GatherTranspose",
            "Transpose",
            [gather_output],
            [gather_reordered],
            { "perm" => perm }
          )
          unsqueeze_input = gather_reordered
        end
        lowered << build_onnx_node_spec(
          "node_#{node_index}_Unsqueeze",
          "Unsqueeze",
          [unsqueeze_input, axes_name],
          outputs,
          {}
        )
        return lowered
      when "Slice"
        starts, ends, axes, steps = slice_vectors_from_arguments(arguments)
        starts_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "starts", starts)
        ends_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "ends", ends)
        axes_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axes", axes)
        steps_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "steps", steps)
        inputs.concat([starts_name, ends_name, axes_name, steps_name])
        inferred_output_shape = infer_slice_output_shape(known_shapes[inputs[0]], starts, ends, axes, steps)
      when "Split"
        axis, lengths = split_axis_and_lengths(arguments, known_shapes[inputs[0]], outputs.length)
        split_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "split", lengths)
        inputs << split_name
        attributes["axis"] = axis

        split_shapes = infer_split_output_shapes(known_shapes[inputs[0]], axis, lengths)
        unless split_shapes.nil?
          outputs.each_with_index do |name, index|
            known_shapes[name] = split_shapes[index]
          end
        end
      when "ArgReduce"
        arg_mode, arg_axis = argreduce_mode_axis(arguments)
        arg_op = ARG_REDUCE_CODE_TO_ONNX_OP.fetch(arg_mode)
        arg_output = unique_aux_tensor_name(used_tensor_names, node_index, "argreduce")
        arg_shape = infer_argreduce_keepdims_shape(known_shapes[inputs[0]], arg_axis)
        known_shapes[arg_output] = arg_shape unless arg_shape.nil?
        outputs.each { |name| known_shapes[name] = arg_shape.dup unless arg_shape.nil? }

        return [
          build_onnx_node_spec(
            "node_#{node_index}_#{arg_op}",
            arg_op,
            inputs,
            [arg_output],
            { "axis" => arg_axis, "keepdims" => 1 }
          ),
          build_onnx_node_spec(
            "node_#{node_index}_CastUint32",
            "Cast",
            [arg_output],
            outputs,
            { "to" => "UINT32" }
          )
        ]
      when "AsStrided"
        input_name = inputs.fetch(0)
        input_shape = known_shapes[input_name]
        unless input_shape
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported AsStrided for tensor #{input_name.inspect} without known static shape"
        end

        output_shape, strides, offset = asstrided_arguments(arguments)
        input_size = tensor_size_from_shape(input_shape, "AsStrided input shape")
        indices = asstrided_linear_indices(output_shape, strides, offset, input_size)
        indices_name = unique_aux_tensor_name(used_tensor_names, node_index, "asstrided_indices")
        initializers << onnx_initializer_info(
          "name" => indices_name,
          "shape" => output_shape,
          "dtype" => "int64",
          "values" => indices
        )

        input_rank = normalize_integer_vector(input_shape, "AsStrided input shape").length
        gather_input = input_name
        lowered = []
        if input_rank != 1
          flatten_shape_name = append_aux_int64_initializer!(
            initializers,
            used_tensor_names,
            node_index,
            "asstrided_flatten_shape",
            [-1]
          )
          gather_input = unique_aux_tensor_name(used_tensor_names, node_index, "asstrided_input_flat")
          lowered << build_onnx_node_spec(
            "node_#{node_index}_AsStridedInputFlatten",
            "Reshape",
            [input_name, flatten_shape_name],
            [gather_input],
            {}
          )
        end

        lowered << build_onnx_node_spec(
          "node_#{node_index}_AsStridedGather",
          "Gather",
          [gather_input, indices_name],
          outputs,
          { "axis" => 0 }
        )
        outputs.each { |name| known_shapes[name] = output_shape.dup }
        return lowered
      when "ScatterAxis"
        inferred_output_shape = known_shapes[inputs[0]]
      when "Greater"
        inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
      when "Select"
        inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[1]], known_shapes[inputs[2]])
      when "Full"
        inferred_output_shape = known_shapes[inputs[0]]
      when "Concatenate"
        axis = concatenate_axis_from_arguments(arguments)
        inferred_output_shape = infer_concatenate_output_shape(inputs.map { |name| known_shapes[name] }, axis)
      when "Flatten"
        flatten_input = inputs.first
        input_shape = known_shapes[flatten_input]
        unless input_shape
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Flatten for tensor #{flatten_input.inspect} without known static shape"
        end
        shape = flatten_shape_from_arguments(arguments, input_shape)
        shape_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "shape", shape)
        inputs << shape_name
        inferred_output_shape = shape
      when "Unflatten"
        unflatten_input = inputs.first
        input_shape = known_shapes[unflatten_input]
        unless input_shape
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Unflatten for tensor #{unflatten_input.inspect} without known static shape"
        end

        shape = unflatten_shape_from_arguments(arguments, input_shape)
        shape_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "shape", shape)
        inputs << shape_name
        inferred_output_shape = shape
      when "Squeeze"
        axes = integer_vector_argument(arguments, "Squeeze")
        axes_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axes", axes)
        inputs << axes_name
        inferred_output_shape = infer_squeeze_output_shape(known_shapes[inputs[0]], axes)
      when "ExpandDims"
        axes = integer_vector_argument(arguments, "ExpandDims")
        axes_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axes", axes)
        inputs << axes_name
        inferred_output_shape = infer_unsqueeze_output_shape(known_shapes[inputs[0]], axes)
      when "Broadcast"
        shape = integer_vector_argument(arguments, "Broadcast")
        shape_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "shape", shape)
        inputs << shape_name
        inferred_output_shape = shape
      end

      unless inferred_output_shape.nil?
        outputs.each do |name|
          known_shapes[name] = inferred_output_shape.dup
        end
      end

      build_onnx_node_spec("node_#{node_index}_#{op_type}", op_type, inputs, outputs, attributes)
    end
    private_class_method :lower_onnx_node

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
        arguments.find { |value| value.is_a?(Array) && value.all? { |item| item.is_a?(Integer) } }
      end
      unless candidate
        raise ArgumentError, "[graph_ir_to_onnx_stub] #{op_name} requires an integer-vector argument"
      end

      candidate
    end
    private_class_method :integer_vector_argument

    def normalize_integer_vector(value, label)
      case value
      when Integer
        [value]
      when Array
        unless value.all? { |item| item.is_a?(Integer) }
          raise TypeError, "[graph_ir_to_onnx_stub] #{label} must contain only Integer values"
        end
        value
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
      unless start.is_a?(Integer) && stop.is_a?(Integer) && step.is_a?(Integer)
        raise TypeError, "[graph_ir_to_onnx_stub] Arange start/stop/step must be Integer"
      end
      raise ArgumentError, "[graph_ir_to_onnx_stub] Arange step must not be zero" if step.zero?

      [start, stop, step]
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
      unless axis.is_a?(Integer)
        raise TypeError, "[graph_ir_to_onnx_stub] #{label} must be an Integer"
      end
      index = axis
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

    def concatenate_axis_from_arguments(arguments, strict: true)
      if arguments.is_a?(Array) && arguments.length == 1 && arguments.first.is_a?(Integer)
        return arguments.first
      end
      return nil unless strict

      raise NotImplementedError,
            "[graph_ir_to_onnx_stub] unsupported Concatenate arguments #{arguments.inspect}; expected [axis]"
    end
    private_class_method :concatenate_axis_from_arguments

    def gather_axis_from_arguments(arguments, strict: true)
      if arguments.is_a?(Array) && !arguments.empty?
        first = arguments.first
        return first if first.is_a?(Integer)
        if first.is_a?(Array) && first.length == 1 && first.first.is_a?(Integer)
          return first.first
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
      unless mode.is_a?(Integer) && axis.is_a?(Integer)
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
