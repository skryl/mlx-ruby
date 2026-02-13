# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase167ModuleStateBehaviorTest < Minitest::Test
  class ToyModule < MLX::NN::Module
    attr_reader :note

    def initialize
      super()
      self.weight = MLX::Core.array([1.0], MLX::Core.float32)
      self.meta = { "depth" => 2 }
      self.axes = [0, 1]
      self.note = "hello"
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_state_tracks_array_and_container_assignments
    mod = ToyModule.new

    assert mod.state.key?("weight")
    assert mod.state.key?("meta")
    assert mod.state.key?("axes")
    refute mod.state.key?("note")

    assert_nested_close [1.0], mod.weight.to_a
    assert_equal({ "depth" => 2 }, mod.meta)
    assert_equal [0, 1], mod.axes
    assert_equal "hello", mod.note
  end

  def test_state_is_reference_and_setter_moves_values_between_state_and_attrs
    mod = ToyModule.new

    mod.state["weight"] = MLX::Core.array([2.0], MLX::Core.float32)
    assert_nested_close [2.0], mod.weight.to_a

    mod.weight = "plain value"
    refute mod.state.key?("weight")
    assert_equal "plain value", mod.weight

    mod.weight = MLX::Core.array([3.0], MLX::Core.float32)
    assert mod.state.key?("weight")
    assert_nested_close [3.0], mod.weight.to_a
  end

  def test_module_values_can_be_nested_in_state
    parent = MLX::NN::Module.new
    child = MLX::NN::Module.new
    child.bias = MLX::Core.array([0.5], MLX::Core.float32)

    parent.child = child

    assert_same child, parent.child
    assert_same child, parent.state["child"]
    assert_nested_close [0.5], parent.child.bias.to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal shape_signature(expected), shape_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |item| flatten(item) }
  end

  def shape_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |item| shape_signature(item) })]
  end
end
