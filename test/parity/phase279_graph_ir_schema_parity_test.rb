# frozen_string_literal: true

require "json"
require "stringio"
require_relative "test_helper"

class Phase279GraphIrSchemaParityTest < Minitest::Test
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

  def test_export_graph_ir_emits_tensor_metadata_and_topology
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end

    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, fun, x, y: y))

    assert_equal 1, payload.fetch("ir_version")
    assert_equal false, payload.fetch("shapeless")

    inputs = payload.fetch("inputs")
    assert_equal 2, inputs.length
    inputs.each do |input|
      assert input.key?("name")
      assert_equal [2], input.fetch("shape")
      assert_equal "float32", input.fetch("dtype")
    end

    keyword_inputs = payload.fetch("keyword_inputs")
    assert_equal 1, keyword_inputs.length
    assert_equal "y", keyword_inputs.first.fetch("name")
    assert_includes inputs.map { |input| input.fetch("name") }, keyword_inputs.first.fetch("tensor")

    outputs = payload.fetch("outputs")
    assert_equal 1, outputs.length
    assert_equal [2], outputs.first.fetch("shape")
    assert_equal "float32", outputs.first.fetch("dtype")

    nodes = payload.fetch("nodes")
    assert_equal %w[Exp Add], nodes.map { |node| node.fetch("op") }
    nodes.each do |node|
      assert node.key?("inputs")
      assert node.key?("outputs")
      assert_kind_of Array, node.fetch("inputs")
      assert_kind_of Array, node.fetch("outputs")
      assert_operator node.fetch("outputs").length, :>, 0
    end

    assert_equal [], payload.fetch("constants")
    assert_equal nodes.last.fetch("outputs"), outputs.map { |out| out.fetch("name") }
  end
end
