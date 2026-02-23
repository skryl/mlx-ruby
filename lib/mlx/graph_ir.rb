# frozen_string_literal: true

require "json"
require_relative "graph_ir/webgpu_harness"

module MLX
  module GraphIR
    IR_VERSION = 1

    module_function

    def export_onnx(
      target_path,
      fun,
      *extras,
      shapeless: false,
      opset: 18,
      model_name: "mlx_graph",
      external_data: false,
      external_data_file: nil,
      external_data_size_threshold: 1024,
      **trace_kwargs
    )
      ensure_native_graph_ir!
      assert_boolean!(shapeless, "shapeless")
      assert_boolean!(external_data, "external_data")
      native_target_path = normalize_binary_target_path!(target_path, "export_onnx")
      native_fun, native_extras, _positional_count, _keyword_names =
        prepare_trace_invocation_without_native_kwargs(fun, extras, trace_kwargs)
      translate_native_unsupported do
        written = MLX::GraphIR::Native.export_onnx(
          native_target_path,
          native_fun,
          native_extras,
          {},
          shapeless,
          opset,
          model_name,
          external_data,
          external_data_file,
          external_data_size_threshold
        )
        unless written.is_a?(String)
          raise TypeError, "MLX::GraphIR::Native.export_onnx must return String path"
        end
        written
      end
    end

    def export_onnx_json(
      fun,
      *extras,
      shapeless: false,
      opset: 18,
      model_name: "mlx_graph",
      **trace_kwargs
    )
      ensure_native_graph_ir!
      assert_boolean!(shapeless, "shapeless")
      native_fun, native_extras, _positional_count, _keyword_names =
        prepare_trace_invocation_without_native_kwargs(fun, extras, trace_kwargs)
      translate_native_unsupported do
        onnx_json = MLX::GraphIR::Native.export_onnx_json(
          native_fun,
          native_extras,
          {},
          shapeless,
          opset,
          model_name
        )
        unless onnx_json.is_a?(String)
          raise TypeError, "MLX::GraphIR::Native.export_onnx_json must return String JSON"
        end
        onnx_json
      end
    end

    def export_onnx_compatibility_report(fun, *extras, shapeless: false, **trace_kwargs)
      ensure_native_graph_ir!
      assert_boolean!(shapeless, "shapeless")
      native_fun, native_extras, _positional_count, _keyword_names =
        prepare_trace_invocation_without_native_kwargs(fun, extras, trace_kwargs)
      report = MLX::GraphIR::Native.export_onnx_compatibility_report(
        native_fun,
        native_extras,
        {},
        shapeless
      )
      unless report.is_a?(Hash)
        raise TypeError, "MLX::GraphIR::Native.export_onnx_compatibility_report must return Hash payload"
      end

      report
    end

    def export_graph_ir(fun, *extras, shapeless: false, **trace_kwargs)
      ensure_native_graph_ir!
      assert_boolean!(shapeless, "shapeless")
      native_fun, native_extras, positional_count, keyword_names =
        prepare_trace_invocation_without_native_kwargs(fun, extras, trace_kwargs)
      payload = MLX::GraphIR::Native.export_graph_ir(native_fun, native_extras, {}, shapeless)
      unless payload.is_a?(Hash)
        raise TypeError, "MLX::GraphIR::Native.export_graph_ir must return Hash payload"
      end
      inject_keyword_inputs!(payload, positional_count, keyword_names)
      payload
    end

    def export_graph_ir_json(fun, *extras, shapeless: false, **trace_kwargs)
      ensure_native_graph_ir!
      assert_boolean!(shapeless, "shapeless")
      native_fun, native_extras, positional_count, keyword_names =
        prepare_trace_invocation_without_native_kwargs(fun, extras, trace_kwargs)
      content = MLX::GraphIR::Native.export_graph_ir_json(native_fun, native_extras, {}, shapeless)
      unless content.is_a?(String)
        raise TypeError, "MLX::GraphIR::Native.export_graph_ir_json must return String JSON"
      end
      return content if keyword_names.empty?

      payload = JSON.parse(content)
      inject_keyword_inputs!(payload, positional_count, keyword_names)
      content = JSON.generate(payload)
      content
    end

    def graph_ir_to_onnx(
      target_path,
      graph_ir_source,
      opset: 18,
      model_name: "mlx_graph",
      external_data: false,
      external_data_file: nil,
      external_data_size_threshold: 1024
    )
      ensure_native_graph_ir!
      assert_boolean!(external_data, "external_data")
      native_target_path = normalize_binary_target_path!(target_path, "graph_ir_to_onnx")
      translate_native_unsupported do
        written = MLX::GraphIR::Native.graph_ir_to_onnx(
          native_target_path,
          graph_ir_source,
          opset,
          model_name,
          external_data,
          external_data_file,
          external_data_size_threshold
        )
        unless written.is_a?(String)
          raise TypeError, "MLX::GraphIR::Native.graph_ir_to_onnx must return String path"
        end
        written
      end
    end

    def graph_ir_to_onnx_json(graph_ir_source, opset: 18, model_name: "mlx_graph")
      ensure_native_graph_ir!
      translate_native_unsupported do
        onnx_json = MLX::GraphIR::Native.graph_ir_to_onnx_json(graph_ir_source, opset, model_name)
        unless onnx_json.is_a?(String)
          raise TypeError, "MLX::GraphIR::Native.graph_ir_to_onnx_json must return String JSON"
        end
        onnx_json
      end
    end

    def ensure_native_graph_ir!
      MLX::Core.ensure_native!
      unless defined?(MLX::GraphIR::Native)
        raise RuntimeError, "MLX::GraphIR::Native is unavailable"
      end
    end
    private_class_method :ensure_native_graph_ir!

    def assert_boolean!(value, label)
      unless value == true || value == false
        raise TypeError, "#{label} must be true or false"
      end
    end
    private_class_method :assert_boolean!

    def translate_native_unsupported
      yield
    rescue MLX::GraphIR::Native::UnsupportedError => e
      raise NotImplementedError, e.message
    end
    private_class_method :translate_native_unsupported

    def normalize_binary_target_path!(target_path, method_name)
      if target_path.respond_to?(:write) && !target_path.respond_to?(:to_path)
        raise ArgumentError, "#{method_name} requires a path-like target, not an IO-like target"
      end

      path = if target_path.respond_to?(:to_path)
        target_path.to_path
      else
        target_path
      end

      unless path.is_a?(String)
        raise TypeError, "target_path must be a String or respond to #to_path"
      end

      path
    end
    private_class_method :normalize_binary_target_path!

    def prepare_trace_invocation_without_native_kwargs(fun, extras, trace_kwargs)
      positional_inputs = extras.dup
      return [fun, positional_inputs, positional_inputs.length, []] if trace_kwargs.empty?

      keyword_entries = trace_kwargs.to_a
      positional_count = positional_inputs.length
      native_fun = lambda do |*all_args|
        positional_args = all_args.first(positional_count)
        keyword_args = {}
        keyword_entries.each_with_index do |(name, _value), idx|
          keyword_args[name] = all_args[positional_count + idx]
        end
        fun.call(*positional_args, **keyword_args)
      end

      keyword_values = keyword_entries.map { |(_name, value)| value }
      keyword_names = keyword_entries.map { |(name, _value)| name.to_s }
      [native_fun, positional_inputs + keyword_values, positional_count, keyword_names]
    end
    private_class_method :prepare_trace_invocation_without_native_kwargs

    def inject_keyword_inputs!(payload, positional_count, keyword_names)
      return payload if keyword_names.empty?
      return payload unless payload.is_a?(Hash)

      inputs = payload["inputs"]
      return payload unless inputs.is_a?(Array)

      keyword_inputs = []
      keyword_names.each_with_index do |name, idx|
        input_entry = inputs[positional_count + idx]
        next unless input_entry.is_a?(Hash)

        tensor = input_entry["name"]
        next unless tensor.is_a?(String) && !tensor.empty?

        keyword_inputs << { "name" => name, "tensor" => tensor }
      end
      payload["keyword_inputs"] = keyword_inputs
      payload
    end
    private_class_method :inject_keyword_inputs!
  end
end
