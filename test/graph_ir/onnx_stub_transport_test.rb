# frozen_string_literal: true

require "json"
require "tmpdir"
require_relative "test_helper"

class Phase336OnnxStubTransportParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_graph_ir_to_onnx_writes_binary_without_public_python_builder_path
    payload = {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [2], "dtype" => "float32" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "z", "shape" => [2], "dtype" => "float32" }],
      "constants" => [
        {
          "name" => "c",
          "shape" => [2],
          "dtype" => "float32",
          "values" => [1.0, -1.0]
        }
      ],
      "nodes" => [{ "op" => "Add", "inputs" => %w[x c], "outputs" => ["z"] }]
    }

    Dir.mktmpdir do |dir|
      target_path = File.join(dir, "graph.onnx")
      written = MLX::GraphIR.graph_ir_to_onnx(target_path, payload, model_name: "phase336")

      assert_equal File.expand_path(target_path), written
      assert File.file?(target_path)
      assert_operator File.size(target_path), :>, 0
    end
  end
end
