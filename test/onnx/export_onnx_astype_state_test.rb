# frozen_string_literal: true

require "json"
require "stringio"
require_relative "test_helper"

class Phase318ExportGraphIrAsTypeStateParityTest < Minitest::Test
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

  def test_export_ir_includes_astype_target_dtype_argument
    fun = lambda do |x|
      MLX::Core.astype(x, MLX::Core.int32)
    end
    x = MLX::Core.array([1.2, 2.3], MLX::Core.float32)

    payload = JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x))
    node = payload.fetch("nodes").find { |n| n.fetch("op") == "AsType" }

    refute_nil node
    assert_equal ["int32"], node.fetch("arguments")
  end
end
