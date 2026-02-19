# frozen_string_literal: true

require "json"
require "stringio"
require_relative "test_helper"

class Phase282GraphIrOnnxStubParityTest < Minitest::Test
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

  def test_graph_ir_to_onnx_stub_maps_core_ops
    stub = MLX::Core.graph_ir_to_onnx_stub(exported_payload, opset: 18, model_name: "exp_add")

    assert_equal "onnx_stub_v1", stub.fetch("format")
    assert_equal 18, stub.fetch("opset")
    graph = stub.fetch("graph")
    assert_equal "exp_add", graph.fetch("name")
    assert_equal %w[Exp Add], graph.fetch("nodes").map { |node| node.fetch("op_type") }
    assert_equal 2, graph.fetch("inputs").length
    assert_equal 1, graph.fetch("outputs").length
  end

  def test_graph_ir_to_onnx_stub_rejects_unknown_ops
    payload = exported_payload
    payload["nodes"][0]["op"] = "UnknownCustomOp"

    error = assert_raises(NotImplementedError) do
      MLX::Core.graph_ir_to_onnx_stub(payload)
    end
    assert_match(/UnknownCustomOp/, error.message)
  end

  private

  def exported_payload
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x, y: y))
  end
end
