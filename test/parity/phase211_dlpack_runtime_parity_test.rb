# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase211DlpackRuntimeParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_dlpack_export_and_roundtrip_contract
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    capsule = x.__dlpack__

    assert_instance_of MLX::Core::DLPackCapsule, capsule
    assert_equal x.shape, capsule.shape
    assert_equal x.dtype, capsule.dtype
    assert_equal x.__dlpack_device, capsule.device

    y = MLX::Core.from_dlpack(capsule)
    assert_instance_of MLX::Core::Array, y
    assert_equal x.to_a, y.to_a
    assert_equal x.dtype, y.dtype
  end

  def test_dlpack_stream_argument_validation
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    assert_instance_of MLX::Core::DLPackCapsule, x.__dlpack__(nil)
    assert_instance_of MLX::Core::DLPackCapsule, x.__dlpack__(0)
    assert_raises(ArgumentError) { x.__dlpack__("bad") }
  end
end
