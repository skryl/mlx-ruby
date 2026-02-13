# frozen_string_literal: true

require_relative "test_helper"

class Phase225ArrayIndexingUpdateEdgeParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_setitem_supports_integer_index_lists
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)
    out = x.__setitem__([0, 2], 9.0)
    assert_equal [9.0, 2.0, 9.0, 4.0], out.to_a
    assert_equal [1.0, 2.0, 3.0, 4.0], x.to_a
  end

  def test_setitem_supports_boolean_masks
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    mask = MLX::Core.array([true, false, true], MLX::Core.bool_)

    out = x.__setitem__(mask, -1.0)
    assert_equal [-1.0, 2.0, -1.0], out.to_a
    assert_equal [1.0, 2.0, 3.0], x.to_a
  end
end
