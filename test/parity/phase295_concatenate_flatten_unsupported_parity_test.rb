# frozen_string_literal: true

require "json"
require "stringio"
require_relative "test_helper"

class Phase295ConcatenateFlattenUnsupportedParityTest < Minitest::Test
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

  def test_flatten_with_unknown_intermediate_shape_is_reported_unsupported
    payload = flatten_after_add_payload
    assert_equal %w[Add Flatten], payload.fetch("nodes").map { |node| node.fetch("op") }
    assert_equal [0, 1], payload.fetch("nodes")[1].fetch("arguments")

    report = MLX::Core.graph_ir_webgpu_compatibility_report(payload)
    assert_equal false, report.fetch("ready_for_stub_conversion")
    assert_equal 1, report.fetch("unsupported_nodes")
    assert_equal ["Flatten"], report.fetch("unsupported_ops")

    error = assert_raises(NotImplementedError) do
      MLX::Core.graph_ir_to_onnx_stub(payload)
    end
    assert_match(/without known static shape/i, error.message)
  end

  private

  def flatten_after_add_payload
    fun = lambda do |x, y|
      summed = MLX::Core.add(x, y)
      MLX::Core.flatten(summed, 0, 1)
    end
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    y = MLX::Core.array([[5.0, 6.0], [7.0, 8.0]], MLX::Core.float32)
    JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x, y))
  end
end
