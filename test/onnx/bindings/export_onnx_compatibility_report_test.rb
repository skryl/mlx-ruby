# frozen_string_literal: true

require_relative "test_helper"

class Phase338ExportOnnxCompatibilityReportParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_onnx_compatibility_report_forwards_trace_inputs_to_native
    singleton = MLX::ONNX::Native.singleton_class
    backup_method = :__phase338_original_export_onnx_compatibility_report
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
          "total_nodes" => 1,
          "unsupported_nodes" => 1,
          "unsupported_ops" => ["FutureCustomFusionOp"],
          "ready_for_stub_conversion" => false
        }
      end
    end

    fun = ->(_x, axis: nil) { axis }
    report = MLX::ONNX.export_onnx_compatibility_report(
      fun,
      :seed,
      shapeless: true,
      axis: 1
    )

    assert_equal "webgpu_compat_report_v1", report.fetch("format")
    assert_equal 1, report.fetch("unsupported_nodes")
    assert_equal ["FutureCustomFusionOp"], report.fetch("unsupported_ops")
    assert_equal false, report.fetch("ready_for_stub_conversion")

    refute_equal fun, captured.fetch(:fun)
    assert_equal [:seed, 1], captured.fetch(:extras)
    assert_equal({}, captured.fetch(:trace_kwargs))
    assert_equal 1, captured.fetch(:fun).call(:seed, 1)
    assert_equal true, captured.fetch(:shapeless)
  ensure
    singleton.class_eval do
      remove_method :export_onnx_compatibility_report if method_defined?(:export_onnx_compatibility_report)
      if method_defined?(backup_method)
        alias_method :export_onnx_compatibility_report, backup_method
        remove_method backup_method
      end
    end
  end

  def test_export_onnx_compatibility_report_requires_boolean_shapeless
    error = assert_raises(TypeError) do
      MLX::ONNX.export_onnx_compatibility_report(-> { nil }, shapeless: nil)
    end
    assert_match("shapeless must be true or false", error.message)
  end
end
