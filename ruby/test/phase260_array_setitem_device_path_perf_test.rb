# frozen_string_literal: true

require_relative "test_helper"

class Phase260ArraySetitemDevicePathPerfTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_setitem_integer_index_on_1d_avoids_to_a
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    x.define_singleton_method(:to_a) do
      raise "__setitem__ integer path should not call to_a"
    end

    out = x.__setitem__(1, 9.0)
    assert_equal [1.0, 9.0, 3.0], out.to_a
  end

  def test_setitem_integer_list_on_1d_avoids_to_a
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)
    x.define_singleton_method(:to_a) do
      raise "__setitem__ integer-list path should not call to_a"
    end

    out = x.__setitem__([0, 2], [7.0, 8.0])
    assert_equal [7.0, 2.0, 8.0, 4.0], out.to_a
  end
end
