# frozen_string_literal: true

require_relative "test_helper"

class Phase88ArrayGetsetitemTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_getitem_and_setitem_surface
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    assert_respond_to x, :__getitem__
    assert_respond_to x, :__setitem__

    assert_in_delta 2.0, x.__getitem__(1).to_a, 1e-6

    updated = x.__setitem__(1, 9.0)
    assert_equal [1.0, 9.0, 3.0], updated.to_a
    assert_equal [1.0, 2.0, 3.0], x.to_a
  end
end
