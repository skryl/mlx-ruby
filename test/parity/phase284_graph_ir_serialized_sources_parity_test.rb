# frozen_string_literal: true

require "json"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase284GraphIrSerializedSourcesParityTest < Minitest::Test
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

  def test_validate_graph_ir_accepts_json_string_and_path
    payload = exported_payload
    json = JSON.generate(payload)
    from_string = MLX::GraphIR.validate!(json)
    assert_equal payload, from_string

    Dir.mktmpdir do |dir|
      path = File.join(dir, "graph_ir.json")
      File.binwrite(path, json)
      from_path = MLX::GraphIR.validate!(path)
      assert_equal payload, from_path
    end
  end

  def test_graph_ir_to_onnx_stub_accepts_json_string_source
    stub = TestSupport.parse_onnx_stub(JSON.generate(exported_payload), model_name: "exp_add")
    assert_equal "onnx_stub_v1", stub.fetch("format")
    assert_equal %w[Exp Add], stub.fetch("graph").fetch("nodes").map { |node| node.fetch("op_type") }
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
