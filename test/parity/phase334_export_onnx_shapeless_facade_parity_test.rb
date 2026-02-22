# frozen_string_literal: true

require "json"
require_relative "test_helper"

class Phase334ExportOnnxShapelessFacadeParityTest < Minitest::Test
  EXPECTED_PUBLIC_METHODS = [
    :compatibility_report,
    :export_graph_ir_json,
    :export_onnx_json,
    :export_onnx_webgpu_harness,
    :graph_ir_to_onnx_json,
    :onnx_json_to_onnx,
    :smoke_test_onnx_webgpu_harness
  ].freeze

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_onnx_json_accepts_and_forwards_shapeless_keyword
    singleton = class << MLX::GraphIR::ONNX::Exporter
      self
    end

    backup_method = :__phase334_original_export_onnx_json
    captured = nil
    singleton.class_eval do
      alias_method backup_method, :export_onnx_json
      define_method(:export_onnx_json) do |fun, *extras, trace_kwargs:, shapeless:, opset:, model_name:|
        captured = {
          fun: fun,
          extras: extras,
          trace_kwargs: trace_kwargs,
          shapeless: shapeless,
          opset: opset,
          model_name: model_name
        }
        JSON.generate(
          "format" => "onnx_stub_v1",
          "ir_version" => 1,
          "opset" => opset,
          "producer_name" => "mlx-ruby",
          "graph" => {
            "name" => model_name,
            "inputs" => [],
            "outputs" => [],
            "initializers" => [],
            "nodes" => []
          }
        )
      end
    end

    fun = ->(_seed) { nil }
    content = MLX::GraphIR.export_onnx_json(
      fun,
      :seed,
      shapeless: true,
      opset: 19,
      model_name: "phase334_model",
      axis: 1
    )
    payload = JSON.parse(content)

    assert_equal "onnx_stub_v1", payload.fetch("format")
    assert_equal true, captured.fetch(:shapeless)
    assert_equal 19, captured.fetch(:opset)
    assert_equal "phase334_model", captured.fetch(:model_name)
    assert_equal [ :seed ], captured.fetch(:extras)
    assert_equal({ axis: 1 }, captured.fetch(:trace_kwargs))
    assert_equal fun, captured.fetch(:fun)
  ensure
    singleton.class_eval do
      remove_method :export_onnx_json if method_defined?(:export_onnx_json)
      if method_defined?(backup_method)
        alias_method :export_onnx_json, backup_method
        remove_method backup_method
      end
    end
  end

  def test_graph_ir_facade_does_not_expose_compatibility_report_json
    refute_includes MLX::GraphIR.public_methods(false), :compatibility_report_json
    assert_raises(NoMethodError) do
      MLX::GraphIR.compatibility_report_json("{}")
    end
  end

  def test_graph_ir_facade_does_not_expose_removed_helpers
    removed = %i[
      compatibility_report_json
      dump_json
      graph_ir_to_onnx_payload
      load_payload
      onnx_json_compatible_value
    ]
    removed.each do |method_name|
      refute_includes MLX::GraphIR.public_methods(false), method_name
      assert_raises(NoMethodError) { MLX::GraphIR.public_send(method_name, "{}") }
    end
  end

  def test_graph_ir_public_method_surface_matches_expected_contract
    public_methods = MLX::GraphIR.public_methods(false)
    expected = EXPECTED_PUBLIC_METHODS + [:validate!]
    expected.each do |method_name|
      assert_includes public_methods, method_name
    end
  end
end
