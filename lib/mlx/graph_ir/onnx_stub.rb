# frozen_string_literal: true

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

    def lower_onnx_node(node, node_index, initializers, used_tensor_names, known_shapes, known_dtypes)
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
      inferred_output_dtype = nil

      case op
      when "Arange"
        start, stop, step, arange_dtype = arange_arguments(arguments)
        values = arange_values(start, stop, step)
        output_name = outputs.fetch(0)
        initializers << onnx_initializer_info(
          "name" => output_name,
          "shape" => [values.length],
          "dtype" => arange_dtype,
          "values" => values
        )
        known_shapes[output_name] = [values.length]
        known_dtypes[output_name] = arange_dtype
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
        inferred_output_dtype = known_dtypes[transpose_input]
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

          input_dtype = known_dtypes[inputs[0]]
          weight_dtype = known_dtypes[inputs[1]]
          conv_output_dtype = promote_binary_dtype(input_dtype, weight_dtype) || input_dtype || weight_dtype
          known_dtypes[transposed_input] = input_dtype unless input_dtype.nil?
          known_dtypes[transposed_weight] = weight_dtype unless weight_dtype.nil?
          unless conv_output_dtype.nil?
            known_dtypes[conv_output] = conv_output_dtype
            outputs.each { |name| known_dtypes[name] = conv_output_dtype }
          end

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
        inferred_output_dtype = known_dtypes[inputs.first]

        if reduce_code == 0 || reduce_code == 1
          cast_bool_out = unique_aux_tensor_name(used_tensor_names, node_index, "cast_bool")
          cast_int_out = unique_aux_tensor_name(used_tensor_names, node_index, "cast_int64")
          reduce_out = unique_aux_tensor_name(used_tensor_names, node_index, "reduce")
          reduce_type = REDUCE_CODE_TO_ONNX_OP.fetch(reduce_code)

          outputs.each do |name|
            known_shapes[name] = inferred_output_shape.dup unless inferred_output_shape.nil?
            known_dtypes[name] = "bool"
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
      when "AsType"
        target_dtype = onnx_effective_dtype(as_type_target_dtype(arguments, outputs, known_dtypes))
        attributes["to"] = ONNX_DTYPE_MAP.fetch(target_dtype)
        inferred_output_shape = known_shapes[inputs[0]]
        inferred_output_dtype = target_dtype
      when "Reshape"
        shape = integer_vector_argument(arguments, "Reshape")
        shape_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "shape", shape)
        inputs << shape_name
        inferred_output_shape = shape
        inferred_output_dtype = known_dtypes[inputs[0]]
      when "Add", "Subtract", "Multiply", "Divide", "Maximum", "Minimum", "Power"
        lhs_dtype = known_dtypes[inputs[0]]
        rhs_dtype = known_dtypes[inputs[1]]
        promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype)
        if promoted_dtype
          inputs, cast_nodes = cast_inputs_to_dtype(
            node_index: node_index,
            op_name: op,
            inputs: inputs,
            target_dtype: promoted_dtype,
            known_shapes: known_shapes,
            known_dtypes: known_dtypes,
            used_tensor_names: used_tensor_names
          )
          unless cast_nodes.empty?
            inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
            inferred_output_dtype = promoted_dtype
            unless inferred_output_shape.nil?
              outputs.each do |name|
                known_shapes[name] = inferred_output_shape.dup
              end
            end
            outputs.each { |name| known_dtypes[name] = inferred_output_dtype } unless inferred_output_dtype.nil?
            cast_nodes << build_onnx_node_spec("node_#{node_index}_#{op_type}", op_type, inputs, outputs, attributes)
            return cast_nodes
          end
        end
        inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
        inferred_output_dtype = promoted_dtype || lhs_dtype || rhs_dtype
      when "Exp", "Log", "Abs", "Negative", "Relu", "Sigmoid", "Tanh", "Softmax", "Sin", "Cos", "Erf", "Floor"
        inferred_output_shape = known_shapes[inputs[0]]
        inferred_output_dtype = known_dtypes[inputs[0]]
      when "Sqrt"
        inferred_output_shape = known_shapes[inputs[0]]
        inferred_output_dtype = known_dtypes[inputs[0]]
        if sqrt_is_reciprocal?(arguments)
          sqrt_output = unique_aux_tensor_name(used_tensor_names, node_index, "sqrt")
          known_shapes[sqrt_output] = inferred_output_shape.dup unless inferred_output_shape.nil?
          known_dtypes[sqrt_output] = inferred_output_dtype unless inferred_output_dtype.nil?
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
        inferred_output_dtype = known_dtypes[inputs[0]]
      when "AddMM"
        alpha, beta = addmm_alpha_beta(arguments)
        attributes["alpha"] = alpha unless alpha == 1.0
        attributes["beta"] = beta unless beta == 1.0
        attributes["transA"] = 0
        attributes["transB"] = 0
        inferred_output_shape = infer_matmul_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
        inferred_output_dtype = known_dtypes[inputs[0]]
      when "Square"
        square_input = inputs.fetch(0)
        inputs = [square_input, square_input]
        inferred_output_shape = known_shapes[square_input]
        inferred_output_dtype = known_dtypes[square_input]
      when "Gather"
        axis = gather_axis_from_arguments(arguments)
        pre_nodes = []
        indices_input = inputs[1]
        indices_dtype = canonical_dtype(known_dtypes[indices_input])
        if !indices_dtype.nil? && !%w[int32 int64].include?(indices_dtype)
          cast_indices = unique_aux_tensor_name(used_tensor_names, node_index, "gather_indices_cast")
          pre_nodes << build_onnx_node_spec(
            "node_#{node_index}_GatherCastIndices",
            "Cast",
            [indices_input],
            [cast_indices],
            { "to" => ONNX_DTYPE_MAP.fetch("int64") }
          )
          index_shape = known_shapes[indices_input]
          known_shapes[cast_indices] = index_shape.dup unless index_shape.nil?
          known_dtypes[cast_indices] = "int64"
          inputs[1] = cast_indices
        end
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
          known_dtypes[gather_output] = known_dtypes[inputs[0]] unless known_dtypes[inputs[0]].nil?
          known_dtypes[gather_reordered] = known_dtypes[inputs[0]] if needs_reorder && !known_dtypes[inputs[0]].nil?
          final_shape = reordered_shape.dup
          final_shape.insert(unsqueeze_axis, 1)
          outputs.each do |name|
            known_shapes[name] = final_shape.dup
            known_dtypes[name] = known_dtypes[inputs[0]] unless known_dtypes[inputs[0]].nil?
          end
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
        return pre_nodes + lowered
      when "GatherAxis"
        axis = gather_axis_from_arguments(arguments)
        pre_nodes = []
        indices_input = inputs[1]
        indices_dtype = canonical_dtype(known_dtypes[indices_input])
        if !indices_dtype.nil? && !%w[int32 int64].include?(indices_dtype)
          cast_indices = unique_aux_tensor_name(used_tensor_names, node_index, "gatheraxis_indices_cast")
          pre_nodes << build_onnx_node_spec(
            "node_#{node_index}_GatherAxisCastIndices",
            "Cast",
            [indices_input],
            [cast_indices],
            { "to" => ONNX_DTYPE_MAP.fetch("int64") }
          )
          index_shape = known_shapes[indices_input]
          known_shapes[cast_indices] = index_shape.dup unless index_shape.nil?
          known_dtypes[cast_indices] = "int64"
          inputs[1] = cast_indices
        end
        data_shape = known_shapes[inputs[0]]
        indices_shape = known_shapes[inputs[1]]
        unless data_shape
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported GatherAxis for tensor #{inputs[0].inspect} without known static shape"
        end
        unless indices_shape
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported GatherAxis for indices #{inputs[1].inspect} without known static shape"
        end
        data_rank = normalize_integer_vector(data_shape, "GatherAxis data shape").length
        indices_rank = normalize_integer_vector(indices_shape, "GatherAxis indices shape").length
        axis_index = normalize_axis(axis, data_rank, "GatherAxis axis")
        unless data_rank == indices_rank
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported GatherAxis rank mismatch: data rank #{data_rank}, indices rank #{indices_rank}"
        end

        data_dims = normalize_integer_vector(data_shape, "GatherAxis data shape")
        indices_dims = normalize_integer_vector(indices_shape, "GatherAxis indices shape")
        expanded_data_shape = data_dims.dup
        needs_data_expand = false
        data_dims.each_with_index do |dim, dim_index|
          next if dim_index == axis_index
          index_dim = indices_dims[dim_index]
          next if dim == index_dim
          if dim == 1
            expanded_data_shape[dim_index] = index_dim
            needs_data_expand = true
            next
          end
          next if index_dim <= dim

          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported GatherAxis shape mismatch at dim #{dim_index}: " \
                "data=#{dim}, indices=#{index_dim}"
        end

        if needs_data_expand
          expand_shape_name = append_aux_int64_initializer!(
            initializers,
            used_tensor_names,
            node_index,
            "gatheraxis_expand_shape",
            expanded_data_shape
          )
          expanded_data = unique_aux_tensor_name(used_tensor_names, node_index, "gatheraxis_expanded_data")
          known_shapes[expanded_data] = expanded_data_shape.dup
          known_dtypes[expanded_data] = known_dtypes[inputs[0]] unless known_dtypes[inputs[0]].nil?

          inferred_output_shape = indices_dims
          inferred_output_dtype = known_dtypes[inputs[0]]
          unless inferred_output_shape.nil?
            outputs.each do |name|
              known_shapes[name] = inferred_output_shape.dup
            end
          end
          unless inferred_output_dtype.nil?
            outputs.each do |name|
              known_dtypes[name] = inferred_output_dtype
            end
          end

          return pre_nodes + [
            build_onnx_node_spec(
              "node_#{node_index}_GatherAxisExpand",
              "Expand",
              [inputs[0], expand_shape_name],
              [expanded_data],
              {}
            ),
            build_onnx_node_spec(
              "node_#{node_index}_#{op_type}",
              op_type,
              [expanded_data, inputs[1]],
              outputs,
              attributes
            )
          ]
        end

        inferred_output_shape = indices_dims
        inferred_output_dtype = known_dtypes[inputs[0]]
        unless pre_nodes.empty?
          unless inferred_output_shape.nil?
            outputs.each do |name|
              known_shapes[name] = inferred_output_shape.dup
            end
          end
          unless inferred_output_dtype.nil?
            outputs.each do |name|
              known_dtypes[name] = inferred_output_dtype
            end
          end
          pre_nodes << build_onnx_node_spec("node_#{node_index}_#{op_type}", op_type, inputs, outputs, attributes)
          return pre_nodes
        end
      when "LogSumExp"
        input_shape = known_shapes[inputs[0]]
        unless input_shape
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported LogSumExp for tensor #{inputs[0].inspect} without known static shape"
        end
        output_shape = known_shapes[outputs.first]
        axes = infer_logsumexp_axes(input_shape, output_shape)
        axes_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axes", axes)
        inputs << axes_name
        attributes["keepdims"] = 1
        inferred_output_shape = infer_reduce_keepdims_shape(input_shape, axes)
        inferred_output_dtype = known_dtypes[inputs[0]]
      when "Pad"
        input_shape = known_shapes[inputs[0]]
        unless input_shape
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Pad for tensor #{inputs[0].inspect} without known static shape"
        end

        axes, pad_low, pad_high = pad_axes_and_sizes_from_arguments(arguments, input_shape)
        rank = normalize_integer_vector(input_shape, "Pad input shape").length
        pads_begin = Array.new(rank, 0)
        pads_end = Array.new(rank, 0)
        axes.each_with_index do |axis, index|
          axis_index = normalize_axis(axis, rank, "Pad axis")
          pads_begin[axis_index] = pad_low[index]
          pads_end[axis_index] = pad_high[index]
        end

        pads_name = append_aux_int64_initializer!(
          initializers,
          used_tensor_names,
          node_index,
          "pads",
          [*pads_begin, *pads_end]
        )

        unless inputs.length.between?(1, 2)
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Pad input arity #{inputs.length}; expected 1 or 2 inputs"
        end
        inputs = [inputs.first, pads_name, *inputs.drop(1)]
        attributes["mode"] = "constant"

        inferred_output_shape = infer_pad_output_shape(input_shape, pads_begin, pads_end)
        inferred_output_dtype = known_dtypes[inputs.first]
      when "Scan"
        reduce_type, axis, reverse, inclusive = scan_arguments(arguments)
        unless reduce_type == 2
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Scan reduce_type #{reduce_type.inspect}; only CumSum (2) is supported"
        end
        axis_value = axis
        unless known_shapes[inputs[0]].nil?
          input_rank = normalize_integer_vector(known_shapes[inputs[0]], "Scan input shape").length
          axis_value = normalize_axis(axis, input_rank, "Scan axis")
        end
        axis_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axis", [axis_value])
        inputs << axis_name
        attributes["exclusive"] = 1 unless inclusive
        attributes["reverse"] = 1 if reverse
        inferred_output_shape = known_shapes[inputs[0]]
        inferred_output_dtype = known_dtypes[outputs.first] || known_dtypes[inputs[0]]
      when "Slice"
        starts, ends, axes, steps = slice_vectors_from_arguments(arguments)
        starts_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "starts", starts)
        ends_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "ends", ends)
        axes_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axes", axes)
        steps_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "steps", steps)
        inputs.concat([starts_name, ends_name, axes_name, steps_name])
        inferred_output_shape = infer_slice_output_shape(known_shapes[inputs[0]], starts, ends, axes, steps)
        inferred_output_dtype = known_dtypes[inputs[0]]
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
        inferred_output_dtype = known_dtypes[inputs[0]]
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
        outputs.each do |name|
          known_shapes[name] = output_shape.dup
          known_dtypes[name] = known_dtypes[input_name] unless known_dtypes[input_name].nil?
        end
        return lowered
      when "ScatterAxis"
        inferred_output_shape = known_shapes[inputs[0]]
        inferred_output_dtype = known_dtypes[inputs[0]]
      when "Greater", "Less"
        lhs_dtype = known_dtypes[inputs[0]]
        rhs_dtype = known_dtypes[inputs[1]]
        promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype)
        if promoted_dtype
          inputs, cast_nodes = cast_inputs_to_dtype(
            node_index: node_index,
            op_name: op,
            inputs: inputs,
            target_dtype: promoted_dtype,
            known_shapes: known_shapes,
            known_dtypes: known_dtypes,
            used_tensor_names: used_tensor_names
          )
          unless cast_nodes.empty?
            inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
            inferred_output_dtype = "bool"
            unless inferred_output_shape.nil?
              outputs.each do |name|
                known_shapes[name] = inferred_output_shape.dup
              end
            end
            outputs.each { |name| known_dtypes[name] = inferred_output_dtype }
            cast_nodes << build_onnx_node_spec("node_#{node_index}_#{op_type}", op_type, inputs, outputs, attributes)
            return cast_nodes
          end
        end
        inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
        inferred_output_dtype = "bool"
      when "Equal"
        equal_nan = equal_nan_from_arguments(arguments)
        if equal_nan
          raise NotImplementedError,
                "[graph_ir_to_onnx_stub] unsupported Equal equal_nan=true; only equal_nan=false is supported"
        end
        lhs_dtype = known_dtypes[inputs[0]]
        rhs_dtype = known_dtypes[inputs[1]]
        promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype)
        if promoted_dtype
          inputs, cast_nodes = cast_inputs_to_dtype(
            node_index: node_index,
            op_name: op,
            inputs: inputs,
            target_dtype: promoted_dtype,
            known_shapes: known_shapes,
            known_dtypes: known_dtypes,
            used_tensor_names: used_tensor_names
          )
          unless cast_nodes.empty?
            inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
            inferred_output_dtype = "bool"
            unless inferred_output_shape.nil?
              outputs.each do |name|
                known_shapes[name] = inferred_output_shape.dup
              end
            end
            outputs.each { |name| known_dtypes[name] = inferred_output_dtype }
            cast_nodes << build_onnx_node_spec("node_#{node_index}_#{op_type}", op_type, inputs, outputs, attributes)
            return cast_nodes
          end
        end
        inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[0]], known_shapes[inputs[1]])
        inferred_output_dtype = "bool"
      when "Select"
        lhs_dtype = known_dtypes[inputs[1]]
        rhs_dtype = known_dtypes[inputs[2]]
        promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype)
        if promoted_dtype
          inputs, cast_nodes = cast_inputs_to_dtype(
            node_index: node_index,
            op_name: op,
            inputs: inputs,
            target_dtype: promoted_dtype,
            known_shapes: known_shapes,
            known_dtypes: known_dtypes,
            used_tensor_names: used_tensor_names,
            indices: [1, 2]
          )
          unless cast_nodes.empty?
            inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[1]], known_shapes[inputs[2]])
            inferred_output_dtype = promoted_dtype
            unless inferred_output_shape.nil?
              outputs.each do |name|
                known_shapes[name] = inferred_output_shape.dup
              end
            end
            outputs.each { |name| known_dtypes[name] = inferred_output_dtype } unless inferred_output_dtype.nil?
            cast_nodes << build_onnx_node_spec("node_#{node_index}_#{op_type}", op_type, inputs, outputs, attributes)
            return cast_nodes
          end
        end
        inferred_output_shape = infer_elementwise_output_shape(known_shapes[inputs[1]], known_shapes[inputs[2]])
        inferred_output_dtype = promoted_dtype || lhs_dtype || rhs_dtype
      when "Full"
        inferred_output_shape = known_shapes[inputs[0]]
        inferred_output_dtype = known_dtypes[inputs[0]]
      when "Concatenate"
        axis = concatenate_axis_from_arguments(arguments)
        inferred_output_shape = infer_concatenate_output_shape(inputs.map { |name| known_shapes[name] }, axis)
        inferred_output_dtype = known_dtypes[inputs.first]
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
        inferred_output_dtype = known_dtypes[flatten_input]
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
        inferred_output_dtype = known_dtypes[unflatten_input]
      when "Squeeze"
        axes = integer_vector_argument(arguments, "Squeeze")
        axes_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axes", axes)
        inputs << axes_name
        inferred_output_shape = infer_squeeze_output_shape(known_shapes[inputs[0]], axes)
        inferred_output_dtype = known_dtypes[inputs[0]]
      when "ExpandDims"
        axes = integer_vector_argument(arguments, "ExpandDims")
        axes_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "axes", axes)
        inputs << axes_name
        inferred_output_shape = infer_unsqueeze_output_shape(known_shapes[inputs[0]], axes)
        inferred_output_dtype = known_dtypes[inputs[0]]
      when "Broadcast"
        shape = integer_vector_argument(arguments, "Broadcast")
        shape_name = append_aux_int64_initializer!(initializers, used_tensor_names, node_index, "shape", shape)
        input_name = inputs.fetch(0)
        input_dtype = known_dtypes[input_name]
        if input_dtype == "bfloat16"
          cast_input = unique_aux_tensor_name(used_tensor_names, node_index, "broadcast_cast_input")
          expand_output = unique_aux_tensor_name(used_tensor_names, node_index, "broadcast_expand_output")
          known_shapes[cast_input] = known_shapes[input_name].dup unless known_shapes[input_name].nil?
          known_shapes[expand_output] = shape.dup
          known_dtypes[cast_input] = "float32"
          known_dtypes[expand_output] = "float32"
          outputs.each do |name|
            known_shapes[name] = shape.dup
            known_dtypes[name] = "bfloat16"
          end

          return [
            build_onnx_node_spec(
              "node_#{node_index}_BroadcastCastInput",
              "Cast",
              [input_name],
              [cast_input],
              { "to" => ONNX_DTYPE_MAP.fetch("float32") }
            ),
            build_onnx_node_spec(
              "node_#{node_index}_Expand",
              "Expand",
              [cast_input, shape_name],
              [expand_output],
              {}
            ),
            build_onnx_node_spec(
              "node_#{node_index}_BroadcastCastOutput",
              "Cast",
              [expand_output],
              outputs,
              { "to" => ONNX_DTYPE_MAP.fetch("bfloat16") }
            )
          ]
        end

        inputs << shape_name
        inferred_output_shape = shape
        inferred_output_dtype = known_dtypes[inputs[0]]
      end

      unless inferred_output_shape.nil?
        outputs.each do |name|
          known_shapes[name] = inferred_output_shape.dup
        end
      end
      unless inferred_output_dtype.nil?
        outputs.each do |name|
          known_dtypes[name] = inferred_output_dtype
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
  end
end
