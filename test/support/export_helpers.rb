# frozen_string_literal: true

module TestSupport
  module ExportHelpers
    def export_graph_ir_to_target(target, fun, *extras, **trace_kwargs)
      content = MLX::ONNX.export_graph_ir_json(fun, *extras, **trace_kwargs)
      if target.respond_to?(:write)
        target.write(content)
        target.rewind if target.respond_to?(:rewind)
        content
      else
        path = target.respond_to?(:to_path) ? target.to_path.to_s : target.to_s
        File.binwrite(path, content)
        nil
      end
    end

    def export_onnx_from_graph_ir_source(
      target,
      payload_or_source,
      opset: 18,
      model_name: "mlx_graph",
      external_data: false,
      external_data_size_threshold: 1024,
      external_data_file: nil
    )
      MLX::ONNX.graph_ir_to_onnx(
        target,
        payload_or_source,
        opset: opset,
        model_name: model_name,
        external_data: external_data,
        external_data_size_threshold: external_data_size_threshold,
        external_data_file: external_data_file
      )
    end

    def export_onnx_direct_from_fun(
      target,
      fun,
      *extras,
      opset: 18,
      model_name: "mlx_graph",
      external_data: false,
      external_data_size_threshold: 1024,
      external_data_file: nil,
      **trace_kwargs
    )
      MLX::ONNX.export_onnx(
        target,
        fun,
        *extras,
        shapeless: false,
        opset: opset,
        model_name: model_name,
        external_data: external_data,
        external_data_size_threshold: external_data_size_threshold,
        external_data_file: external_data_file,
        **trace_kwargs
      )
    end

    def export_onnx_json_dump(target, payload_or_source, opset: 18, model_name: "mlx_graph", pretty: true)
      onnx_json = MLX::ONNX.graph_ir_to_onnx_json(
        payload_or_source,
        opset: opset,
        model_name: model_name
      )
      content = pretty ? JSON.pretty_generate(JSON.parse(onnx_json)) : onnx_json
      if target.respond_to?(:write)
        target.write(content)
        target.rewind if target.respond_to?(:rewind)
        content
      else
        path = target.respond_to?(:to_path) ? target.to_path.to_s : target.to_s
        File.binwrite(path, content)
        nil
      end
    end

    def parse_onnx_stub(payload_or_source, opset: 18, model_name: "mlx_graph")
      content = MLX::ONNX.graph_ir_to_onnx_json(
        payload_or_source,
        opset: opset,
        model_name: model_name
      )
      payload = JSON.parse(content)
      unless payload.is_a?(Hash)
        raise TypeError, "graph_ir_to_onnx_json must return a JSON object payload"
      end
      payload
    end
  end
end
