# frozen_string_literal: true

require "json"
require "stringio"
require_relative "test_helper"

class Phase289GraphIrWebgpuCompatReportParityTest < Minitest::Test
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

  def test_webgpu_compatibility_report_for_supported_graph
    report = MLX::GraphIR.compatibility_report(exported_payload)
    assert_equal "webgpu_compat_report_v1", report.fetch("format")
    assert_equal 2, report.fetch("total_nodes")
    assert_equal 0, report.fetch("unsupported_nodes")
    assert_equal [], report.fetch("unsupported_ops")
    assert_equal true, report.fetch("ready_for_stub_conversion")
  end

  def test_webgpu_compatibility_report_lists_unsupported_ops
    payload = exported_payload
    payload["nodes"][0]["op"] = "FutureCustomFusionOp"

    report = MLX::GraphIR.compatibility_report(payload)
    assert_equal 1, report.fetch("unsupported_nodes")
    assert_equal ["FutureCustomFusionOp"], report.fetch("unsupported_ops")
    assert_equal false, report.fetch("ready_for_stub_conversion")
  end

  private

  def exported_payload
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x, y: y))
  end
end
