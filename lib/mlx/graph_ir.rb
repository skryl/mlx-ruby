# frozen_string_literal: true

require "json"
require "set"

require_relative "graph_ir/constants"
require_relative "graph_ir/payload"
require_relative "graph_ir/validation"
require_relative "graph_ir/exporter"
require_relative "graph_ir/onnx/python_builder"
require_relative "graph_ir/onnx/exporter"
require_relative "graph_ir/webgpu_harness"

module MLX
  module GraphIR
    module_function

    def export_graph_ir_json(fun, *extras, **trace_kwargs)
      Exporter.export_graph_ir_json(fun, *extras, **trace_kwargs)
    end

    def graph_ir_to_onnx_json(payload_or_source, opset: 18, model_name: "mlx_graph")
      ONNX::Exporter.graph_ir_to_onnx_json(
        payload_or_source,
        opset: opset,
        model_name: model_name
      )
    end

    def graph_ir_to_onnx_payload(payload_or_source, opset: 18, model_name: "mlx_graph")
      payload = JSON.parse(
        graph_ir_to_onnx_json(
          payload_or_source,
          opset: opset,
          model_name: model_name
        )
      )
      unless payload.is_a?(Hash)
        raise TypeError, "graph_ir_to_onnx_json must return a JSON object payload"
      end
      payload
    end

    def export_onnx_json(fun, *extras, opset: 18, model_name: "mlx_graph", **trace_kwargs)
      ONNX::Exporter.export_onnx_json(
        fun,
        *extras,
        trace_kwargs: trace_kwargs,
        opset: opset,
        model_name: model_name
      )
    end

    def compatibility_report(payload_or_source)
      ONNX::Exporter.compatibility_report(payload_or_source)
    end

    def compatibility_report_json(payload_or_source)
      ONNX::Exporter.compatibility_report_json(payload_or_source)
    end

    def onnx_json_to_onnx(
      target,
      onnx_json,
      external_data: false,
      external_data_size_threshold: 1024,
      external_data_file: nil,
      python_bin: ENV.fetch("PYTHON", "python3")
    )
      ONNX::Exporter.onnx_json_to_onnx(
        target,
        onnx_json,
        external_data: external_data,
        external_data_size_threshold: external_data_size_threshold,
        external_data_file: external_data_file,
        python_bin: python_bin
      )
    end

    def export_onnx_webgpu_harness(
      target_dir,
      payload_or_source,
      opset: 18,
      model_name: "mlx_graph",
      execution_providers: %w[webgpu wasm],
      benchmark_warmup_runs: 2,
      benchmark_measure_runs: 10,
      external_data: false,
      external_data_size_threshold: 1024,
      external_data_file: nil
    )
      WebGPUHarness.export_onnx_webgpu_harness(
        target_dir,
        payload_or_source,
        opset: opset,
        model_name: model_name,
        execution_providers: execution_providers,
        benchmark_warmup_runs: benchmark_warmup_runs,
        benchmark_measure_runs: benchmark_measure_runs,
        external_data: external_data,
        external_data_size_threshold: external_data_size_threshold,
        external_data_file: external_data_file
      )
    end

    def smoke_test_onnx_webgpu_harness(
      harness_dir,
      timeout_seconds: 30,
      mock_ort: false,
      local_ort: true,
      node_bin: ENV.fetch("NODE", "node")
    )
      WebGPUHarness.smoke_test_onnx_webgpu_harness(
        harness_dir,
        timeout_seconds: timeout_seconds,
        mock_ort: mock_ort,
        local_ort: local_ort,
        node_bin: node_bin
      )
    end
  end
end
