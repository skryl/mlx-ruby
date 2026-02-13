# frozen_string_literal: true

require_relative "test_helper"

class Phase224ArrayBufferProtocolParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_dlpack_capsule_roundtrip_survives_source_release
    source = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    capsule = source.__dlpack__
    restored = MLX::Core.from_dlpack(capsule)

    source = nil
    GC.start

    assert_equal [[1.0, 2.0], [3.0, 4.0]], restored.to_a
    assert_equal MLX::Core.float32, restored.dtype
  end

  def test_dlpack_device_tuple_contract
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    device = x.__dlpack_device__
    assert_equal 2, device.length
    assert device[0].is_a?(Integer)
    assert device[1].is_a?(Integer)
  end
end
