# frozen_string_literal: true

require_relative "test_helper"

class Phase247VmapStabilityTypeParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_vmap_concatenate_split_flatten_and_conv
    mats = MLX::Core.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [0.0, 1.0]]], MLX::Core.float32)

    concatenate = MLX::Core.vmap(->(m) { MLX::Core.concatenate([m, m], 0) })
    assert_equal [2, 4, 2], concatenate.call(mats).shape

    split = MLX::Core.vmap(->(m) { MLX::Core.split(m, 2, 0) })
    split_out = split.call(mats)
    assert_equal 2, split_out.length
    assert_equal [2, 1, 2], split_out[0].shape
    assert_equal [2, 1, 2], split_out[1].shape

    flatten = MLX::Core.vmap(->(m) { MLX::Core.flatten(m) })
    assert_equal [2, 4], flatten.call(mats).shape

    conv = MLX::Core.vmap(->(t) { MLX::Core.conv2d(t, MLX::Core.array([[[[1.0]]]], MLX::Core.float32)) })
    image_batch = MLX::Core.array([[[[[1.0]]]], [[[[2.0]]]]], MLX::Core.float32)
    assert_equal [2, 1, 1, 1, 1], conv.call(image_batch).shape
  end

  def test_vmap_dtype_stability_over_repeated_calls
    vmapped = MLX::Core.vmap(->(m) { MLX::Core.add(m, 1.0) })
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)

    50.times do
      out = vmapped.call(x)
      assert_equal MLX::Core.float32, out.dtype
      assert_equal [[2.0, 3.0], [4.0, 5.0]], out.to_a
    end
  end
end
