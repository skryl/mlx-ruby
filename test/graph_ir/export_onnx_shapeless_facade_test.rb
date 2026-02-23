# frozen_string_literal: true

require "json"
require_relative "test_helper"

class Phase334ExportOnnxShapelessFacadeParityTest < Minitest::Test
  EXPECTED_PUBLIC_METHODS = [
    :export_onnx,
    :export_onnx_json,
    :export_onnx_compatibility_report,
    :export_graph_ir,
    :export_graph_ir_json,
    :graph_ir_to_onnx,
    :graph_ir_to_onnx_json
  ].freeze

  REMOVED_PUBLIC_METHODS = [
    :compatibility_report,
    :onnx_json_to_onnx,
    :validate!,
    :export_onnx_webgpu_harness,
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
    singleton = MLX::GraphIR::Native.singleton_class
    backup_method = :__phase334_original_export_onnx_json
    captured = nil

    singleton.class_eval do
      alias_method backup_method, :export_onnx_json
      define_method(:export_onnx_json) do |fun, extras, trace_kwargs, shapeless, opset, model_name|
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

    fun = ->(_seed, axis: nil) { axis }
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
    assert_equal [:seed, 1], captured.fetch(:extras)
    assert_equal({}, captured.fetch(:trace_kwargs))
    refute_equal fun, captured.fetch(:fun)
    assert_equal 1, captured.fetch(:fun).call(:seed, 1)
  ensure
    singleton.class_eval do
      remove_method :export_onnx_json if method_defined?(:export_onnx_json)
      if method_defined?(backup_method)
        alias_method :export_onnx_json, backup_method
        remove_method backup_method
      end
    end
  end

  def test_export_onnx_compatibility_report_accepts_and_forwards_shapeless_keyword
    singleton = MLX::GraphIR::Native.singleton_class
    backup_method = :__phase334_original_export_onnx_compatibility_report
    captured = nil

    singleton.class_eval do
      alias_method backup_method, :export_onnx_compatibility_report
      define_method(:export_onnx_compatibility_report) do |fun, extras, trace_kwargs, shapeless|
        captured = {
          fun: fun,
          extras: extras,
          trace_kwargs: trace_kwargs,
          shapeless: shapeless
        }
        {
          "format" => "webgpu_compat_report_v1",
          "total_nodes" => 0,
          "unsupported_nodes" => 0,
          "unsupported_ops" => [],
          "ready_for_stub_conversion" => true
        }
      end
    end

    fun = ->(_seed, axis: nil) { axis }
    report = MLX::GraphIR.export_onnx_compatibility_report(
      fun,
      :seed,
      shapeless: true,
      axis: 1
    )

    assert_equal "webgpu_compat_report_v1", report.fetch("format")
    assert_equal true, captured.fetch(:shapeless)
    assert_equal [:seed, 1], captured.fetch(:extras)
    assert_equal({}, captured.fetch(:trace_kwargs))
    refute_equal fun, captured.fetch(:fun)
    assert_equal 1, captured.fetch(:fun).call(:seed, 1)
  ensure
    singleton.class_eval do
      remove_method :export_onnx_compatibility_report if method_defined?(:export_onnx_compatibility_report)
      if method_defined?(backup_method)
        alias_method :export_onnx_compatibility_report, backup_method
        remove_method backup_method
      end
    end
  end

  def test_graph_ir_facade_does_not_expose_removed_methods
    public_methods = MLX::GraphIR.singleton_methods(false)
    REMOVED_PUBLIC_METHODS.each do |method_name|
      refute_includes public_methods, method_name
      assert_raises(NoMethodError) { MLX::GraphIR.public_send(method_name) }
    end
  end

  def test_graph_ir_public_method_surface_matches_expected_contract_exactly
    assert_equal EXPECTED_PUBLIC_METHODS.sort, MLX::GraphIR.singleton_methods(false).sort
  end
end
