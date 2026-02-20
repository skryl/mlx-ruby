# frozen_string_literal: true

require "json"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase333ExportOnnxDirectParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_onnx_direct_matches_export_onnx_from_exported_payload
    fun = lambda do |x, y:|
      MLX::Core.add(x, y)
    end
    x = MLX::Core.array([[1.0, 2.0]], MLX::Core.float32)
    y = MLX::Core.array([[3.0, 4.0]], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      direct_path = File.join(dir, "direct.onnx")
      payload_path = File.join(dir, "payload.onnx")

      assert_nil TestSupport.export_onnx_direct_from_fun(
        direct_path,
        fun,
        x,
        y: y,
        model_name: "phase333_direct"
      )

      payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x, y: y))
      assert_nil TestSupport.export_onnx_from_graph_ir_source(
        payload_path,
        payload,
        model_name: "phase333_direct"
      )

      direct = File.binread(direct_path)
      via_payload = File.binread(payload_path)
      assert_operator direct.bytesize, :>, 0
      assert_equal via_payload, direct
    end
  end
end
