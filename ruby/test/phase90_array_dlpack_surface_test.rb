# frozen_string_literal: true

require_relative "test_helper"

class Phase90ArrayDlpackSurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_dlpack_surface_methods_exist
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    assert_respond_to x, :__dlpack__
    assert_respond_to x, :__dlpack_device

    device = x.__dlpack_device
    assert_kind_of Array, device
    assert_equal 2, device.length
    assert_includes [1, 8, 13, :cpu, :gpu], device[0]
    assert_kind_of Integer, device[1]

    capsule = x.__dlpack__
    assert_instance_of MLX::Core::DLPackCapsule, capsule
    assert_equal x.shape, capsule.shape
    assert_equal x.dtype, capsule.dtype
  end
end
