# frozen_string_literal: true

require "json"
require_relative "test_helper"

class Phase281GraphIrValidationParityTest < Minitest::Test
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

  def test_export_graph_ir_hash_matches_json_export_payload
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    from_hash = MLX::GraphIR.export_graph_ir(fun, x, y: y)
    from_json = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x, y: y))

    assert_equal from_json, from_hash
  end

  def test_graph_ir_to_onnx_json_accepts_exported_payload
    payload = exported_payload
    stub = JSON.parse(MLX::GraphIR.graph_ir_to_onnx_json(payload, model_name: "phase281_case"))
    assert_equal "onnx_stub_v1", stub.fetch("format")
    assert_equal "phase281_case", stub.fetch("graph").fetch("name")
  end

  def test_graph_ir_to_onnx_json_rejects_malformed_payload
    payload = exported_payload
    payload["constants"] = [{ "name" => "c", "shape" => [1], "dtype" => "float32" }]

    error = assert_raises(RuntimeError) do
      MLX::GraphIR.graph_ir_to_onnx_json(payload)
    end
    assert_match(/values|unsupported|missing/i, error.message)
  end

  private

  def exported_payload
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    MLX::GraphIR.export_graph_ir(fun, x, y: y)
  end
end
