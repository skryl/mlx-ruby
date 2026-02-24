# frozen_string_literal: true

require "json"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase329GraphIrFileReadFallbackParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_ir_to_onnx_json_accepts_path_source
    payload = ir_payload
    Dir.mktmpdir do |dir|
      path = File.join(dir, "ir.json")
      File.binwrite(path, JSON.generate(payload))

      stub = JSON.parse(MLX::ONNX.graph_ir_to_onnx_json(path, model_name: "phase329_path_source"))
      assert_equal "onnx_stub_v1", stub.fetch("format")
      assert_equal "phase329_path_source", stub.fetch("graph").fetch("name")
    end
  end

  def test_ir_to_onnx_json_accepts_io_source
    io = StringIO.new(JSON.generate(ir_payload))
    stub = JSON.parse(MLX::ONNX.graph_ir_to_onnx_json(io, model_name: "phase329_io_source"))
    assert_equal "onnx_stub_v1", stub.fetch("format")
    assert_equal "phase329_io_source", stub.fetch("graph").fetch("name")
  end

  private

  def ir_payload
    {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [
        { "name" => "A", "shape" => [2], "dtype" => "float32" }
      ],
      "keyword_inputs" => [],
      "outputs" => [
        { "name" => "C", "shape" => [2], "dtype" => "float32" }
      ],
      "constants" => [
        { "name" => "B", "shape" => [2], "dtype" => "float32", "values" => [1.0, 2.0] }
      ],
      "nodes" => [
        { "op" => "Add", "inputs" => ["A", "B"], "outputs" => ["C"], "arguments" => [] }
      ]
    }
  end
end
