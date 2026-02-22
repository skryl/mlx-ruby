# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase333ExportOnnxDirectParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    skip "python onnx module is required for phase333 tests" unless python_module_available?("onnx")
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

      direct_written = TestSupport.export_onnx_direct_from_fun(
        direct_path,
        fun,
        x,
        y: y,
        model_name: "phase333_direct"
      )
      assert_equal direct_path, direct_written

      payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, x, y: y))
      payload_written = TestSupport.export_onnx_from_graph_ir_source(
        payload_path,
        payload,
        model_name: "phase333_direct"
      )
      assert_equal payload_path, payload_written

      direct = File.binread(direct_path)
      via_payload = File.binread(payload_path)
      assert_operator direct.bytesize, :>, 0
      assert_equal via_payload, direct
    end
  end

  private

  def python_module_available?(name)
    python_bin = ENV.fetch("PYTHON", "python3")
    _out, _err, status = Open3.capture3(python_bin, "-c", "import #{name}")
    status.success?
  rescue Errno::ENOENT
    false
  end
end
