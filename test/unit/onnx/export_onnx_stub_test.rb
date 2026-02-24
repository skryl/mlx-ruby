# frozen_string_literal: true

require "json"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase283ExportOnnxStubParityTest < Minitest::Test
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

  def test_export_onnx_stub_accepts_file_like_targets
    io = StringIO.new
    written = TestSupport.export_onnx_json_dump(io, exported_payload, model_name: "exp_add")
    payload = JSON.parse(written)

    assert_equal "onnx_stub_v1", payload.fetch("format")
    assert_equal "exp_add", payload.fetch("graph").fetch("name")
    assert_operator io.string.bytesize, :>, 0
  end

  def test_export_onnx_stub_supports_path_targets
    Dir.mktmpdir do |dir|
      path = File.join(dir, "graph.onnx.json")
      assert_nil TestSupport.export_onnx_json_dump(path, exported_payload, model_name: "exp_add")

      assert File.exist?(path)
      payload = JSON.parse(File.read(path))
      assert_equal "onnx_stub_v1", payload.fetch("format")
      assert_equal %w[Exp Add], payload.fetch("graph").fetch("nodes").map { |n| n.fetch("op_type") }
    end
  end

  private

  def exported_payload
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x, y: y))
  end
end
