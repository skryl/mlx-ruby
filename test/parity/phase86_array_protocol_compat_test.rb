# frozen_string_literal: true

require_relative "test_helper"

class Phase86ArrayProtocolCompatTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_python_style_protocol_methods_exist_and_work
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    scalar = MLX::Core.array(2.0, MLX::Core.float32)

    %i[__repr__ __bool__ __int__ __float__ __hash__ __array_namespace__ __eq__ __ne__].each do |name|
      assert_respond_to x, name
    end

    assert_kind_of String, x.__repr__
    assert_kind_of Integer, x.__hash__
    assert_equal MLX::Core, x.__array_namespace__

    assert_equal true, scalar.__bool__
    assert_equal 2, scalar.__int__
    assert_in_delta 2.0, scalar.__float__, 1e-6

    eq = x.__eq__(x)
    ne = x.__ne__(x)
    assert_equal [true, true], eq.to_a
    assert_equal [false, false], ne.to_a
  end
end
