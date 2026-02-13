# frozen_string_literal: true

require_relative "test_helper"

class Phase274SetitemMlxIndexFastPathPerfTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_setitem_mlx_integer_index_vector_avoids_to_a_fallback
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)
    idx = MLX::Core.array([0, 2], MLX::Core.int32)

    x.define_singleton_method(:to_a) { raise "__setitem__ mlx-int index path should not call to_a" }
    idx.define_singleton_method(:to_a) { raise "__setitem__ mlx-int index path should not call index.to_a" }

    out = x.__setitem__(idx, [9.0, 8.0])
    assert_equal [9.0, 2.0, 8.0, 4.0], out.to_a
  end

  def test_setitem_mlx_boolean_mask_vector_avoids_to_a_fallback
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)
    mask = MLX::Core.array([true, false, true, false], MLX::Core.bool_)

    x.define_singleton_method(:to_a) { raise "__setitem__ mlx-bool mask path should not call to_a" }
    mask.define_singleton_method(:to_a) { raise "__setitem__ mlx-bool mask path should not call mask.to_a" }

    out = x.__setitem__(mask, 5.0)
    assert_equal [5.0, 2.0, 5.0, 4.0], out.to_a
  end
end
