# frozen_string_literal: true

require_relative "test_helper"

class Phase7IndexConcatStackSplitTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_slice_and_array_indexing
    x = MLX::Core.reshape(MLX::Core.arange(0, 9, 1, MLX::Core.float32), [3, 3])

    window = MLX::Core.slice(x, [1, 0], [2, 3])
    assert_nested_close [[3.0, 4.0, 5.0]], window.to_a

    row = x[1]
    assert_instance_of MLX::Core::Array, row
    assert_nested_close [3.0, 4.0, 5.0], row.to_a
  end

  def test_concatenate_stack_and_split
    a = MLX::Core.reshape(MLX::Core.arange(0, 4, 1, MLX::Core.float32), [2, 2])
    b = MLX::Core.reshape(MLX::Core.arange(4, 8, 1, MLX::Core.float32), [2, 2])

    concat0 = MLX::Core.concatenate([a, b], 0)
    assert_equal [4, 2], concat0.shape
    assert_nested_close [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], concat0.to_a

    concat1 = MLX::Core.concatenate([a, b], 1)
    assert_equal [2, 4], concat1.shape
    assert_nested_close [[0.0, 1.0, 4.0, 5.0], [2.0, 3.0, 6.0, 7.0]], concat1.to_a

    stacked = MLX::Core.stack([a, b], 0)
    assert_equal [2, 2, 2], stacked.shape
    assert_nested_close [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], stacked.to_a

    splits = MLX::Core.split(concat0, 2, 0)
    assert_equal 2, splits.length
    assert_nested_close a.to_a, splits[0].to_a
    assert_nested_close b.to_a, splits[1].to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal structure_signature(expected), structure_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |x| flatten(x) }
  end

  def structure_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| structure_signature(v) })]
  end
end
