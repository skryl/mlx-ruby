# frozen_string_literal: true

module MLX
  module GraphIR
    module_function

    def validate!(payload_or_source)
      payload = load_payload(payload_or_source)
      validate_top_level!(payload)
      validate_graph_topology!(payload)
      payload
    end

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
  end
end
