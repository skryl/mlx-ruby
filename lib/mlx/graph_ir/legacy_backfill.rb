# frozen_string_literal: true

module MLX
  module GraphIR
    module_function

    LEGACY_GRAPH_IR_DTYPE_PRESERVING_UNARY_OPS = %w[
      Exp
      Log
      Sin
      Cos
      Erf
      Sqrt
      Abs
      Floor
      Negative
      Relu
      Sigmoid
      Tanh
      Softmax
      Reshape
      Flatten
      Unflatten
      Transpose
      Squeeze
      ExpandDims
      Broadcast
      Pad
      Slice
      Scan
      LogSumExp
    ].freeze

    LEGACY_GRAPH_IR_DTYPE_UNIFORM_OPS = %w[
      Add
      Subtract
      Multiply
      Divide
      Maximum
      Minimum
      Power
      Matmul
      AddMM
      Convolution
      ConvolutionTranspose
    ].freeze

    def normalize_exported_graph_ir_file!(path, include_content: true)
      if skip_graph_ir_normalization?(path)
        return read_file_with_fallback(path) if include_content

        return nil
      end

      content = read_file_with_fallback(path)
      normalized = normalize_exported_graph_ir_content(content)
      if normalized.nil?
        return content if include_content

        return nil
      end

      File.binwrite(path, normalized)
      include_content ? normalized : nil
    end

    def skip_graph_ir_normalization?(path)
      File.size(path) > GRAPH_IR_NORMALIZATION_MAX_BYTES
    rescue Errno::ENOENT
      false
    end
    private_class_method :skip_graph_ir_normalization?

    def normalize_exported_graph_ir_content(content)
      payload = JSON.parse(content)
      return nil unless backfill_legacy_graph_ir_dtype_arguments!(payload)

      JSON.generate(payload)
    rescue JSON::ParserError
      nil
    end
    private_class_method :normalize_exported_graph_ir_content

    def backfill_legacy_graph_ir_dtype_arguments!(payload)
      nodes = payload["nodes"]
      return false unless nodes.is_a?(::Array)

      dtype_by_tensor = infer_graph_ir_tensor_dtypes(payload)
      changed = false
      nodes.each do |node|
        next unless node.is_a?(Hash)

        case node["op"]
        when "AsType"
          changed = true if backfill_legacy_astype_arguments!(node, dtype_by_tensor)
        when "NumberOfElements"
          changed = true if backfill_legacy_number_of_elements_arguments!(node, dtype_by_tensor)
        end
      end
      changed
    end
    private_class_method :backfill_legacy_graph_ir_dtype_arguments!

    def backfill_legacy_astype_arguments!(node, dtype_by_tensor)
      arguments = node["arguments"]
      if arguments.is_a?(::Array) && arguments.first.is_a?(String) && !arguments.first.empty?
        return false
      end

      output_dtype = unique_known_dtype(dtype_by_tensor, node["outputs"])
      return false if output_dtype.nil?

      node["arguments"] = [output_dtype]
      true
    end
    private_class_method :backfill_legacy_astype_arguments!

    def backfill_legacy_number_of_elements_arguments!(node, dtype_by_tensor)
      arguments = node["arguments"]
      return false unless arguments.is_a?(::Array) && arguments.length == 2

      output_dtype = unique_known_dtype(dtype_by_tensor, node["outputs"])
      return false if output_dtype.nil?

      node["arguments"] = [arguments[0], arguments[1], output_dtype]
      true
    end
    private_class_method :backfill_legacy_number_of_elements_arguments!

    def infer_graph_ir_tensor_dtypes(payload)
      known = extract_graph_ir_tensor_dtypes(payload)
      nodes = payload["nodes"]
      return known unless nodes.is_a?(::Array)

      loop do
        changed = false
        nodes.each do |node|
          next unless node.is_a?(Hash)

          inputs = node["inputs"].is_a?(::Array) ? node["inputs"] : []
          outputs = node["outputs"].is_a?(::Array) ? node["outputs"] : []
          case node["op"]
          when "AsType"
            arguments = node["arguments"]
            target = arguments.is_a?(::Array) ? arguments.first : nil
            next unless target.is_a?(String) && !target.empty?

            outputs.each do |name|
              changed = true if assign_inferred_dtype!(known, name, target)
            end
          when "Equal", "Greater", "Less"
            outputs.each do |name|
              changed = true if assign_inferred_dtype!(known, name, "bool")
            end
          when "ArgReduce"
            outputs.each do |name|
              changed = true if assign_inferred_dtype!(known, name, "int64")
            end
          when "AsStrided", "Gather", "GatherAxis", "Split"
            changed = true if propagate_primary_input_dtype!(known, inputs, outputs)
          when "Concatenate"
            changed = true if propagate_uniform_dtype_between_tensors!(known, inputs, outputs)
          when "Select"
            changed = true if propagate_uniform_dtype_between_tensors!(known, inputs.drop(1), outputs)
            changed = true if assign_inferred_dtype!(known, inputs.first, "bool")
          when *LEGACY_GRAPH_IR_DTYPE_PRESERVING_UNARY_OPS
            changed = true if propagate_primary_input_dtype!(known, inputs, outputs)
          when *LEGACY_GRAPH_IR_DTYPE_UNIFORM_OPS
            changed = true if propagate_uniform_dtype_between_tensors!(known, inputs, outputs)
          end
        end
        break unless changed
      end

      known
    end
    private_class_method :infer_graph_ir_tensor_dtypes

    def extract_graph_ir_tensor_dtypes(payload)
      %w[inputs outputs constants].each_with_object({}) do |section, out|
        tensors = payload[section]
        next unless tensors.is_a?(::Array)

        tensors.each do |tensor|
          next unless tensor.is_a?(Hash)

          name = tensor["name"]
          dtype = tensor["dtype"]
          next unless name.is_a?(String) && !name.empty?
          next unless dtype.is_a?(String) && !dtype.empty?

          out[name] = dtype
        end
      end
    end
    private_class_method :extract_graph_ir_tensor_dtypes

    def propagate_primary_input_dtype!(known, inputs, outputs)
      return false unless inputs.is_a?(::Array) && outputs.is_a?(::Array)
      input_name = inputs.first
      return false unless input_name.is_a?(String) && !input_name.empty?

      changed = false
      input_dtype = known[input_name]
      output_dtype = unique_known_dtype(known, outputs)

      if input_dtype.nil?
        changed = true if assign_inferred_dtype!(known, input_name, output_dtype)
      else
        outputs.each do |name|
          changed = true if assign_inferred_dtype!(known, name, input_dtype)
        end
      end
      changed
    end
    private_class_method :propagate_primary_input_dtype!

    def propagate_uniform_dtype_between_tensors!(known, left_names, right_names)
      return false unless left_names.is_a?(::Array) && right_names.is_a?(::Array)

      left_dtype = unique_known_dtype(known, left_names)
      right_dtype = unique_known_dtype(known, right_names)
      dtype = left_dtype || right_dtype
      return false if dtype.nil?

      changed = false
      left_names.each do |name|
        changed = true if assign_inferred_dtype!(known, name, dtype)
      end
      right_names.each do |name|
        changed = true if assign_inferred_dtype!(known, name, dtype)
      end
      changed
    end
    private_class_method :propagate_uniform_dtype_between_tensors!

    def unique_known_dtype(known, names)
      return nil unless names.is_a?(::Array)

      known_values = names.filter_map do |name|
        name.is_a?(String) && !name.empty? ? known[name] : nil
      end.uniq
      return nil unless known_values.length == 1

      known_values.first
    end
    private_class_method :unique_known_dtype

    def assign_inferred_dtype!(known, tensor_name, dtype)
      return false unless tensor_name.is_a?(String) && !tensor_name.empty?
      return false unless dtype.is_a?(String) && !dtype.empty?

      existing = known[tensor_name]
      return false if existing == dtype
      return false unless existing.nil?

      known[tensor_name] = dtype
      true
    end
    private_class_method :assign_inferred_dtype!
  end
end
