# frozen_string_literal: true

require_relative "test_helper"

class Phase256ClipGradNormDeviceOnlyPerfTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_clip_grad_norm_does_not_materialize_gradients_to_host
    grad = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    grad.define_singleton_method(:to_a) do
      raise "clip_grad_norm should not call to_a"
    end

    clipped, total_norm = MLX::Optimizers.clip_grad_norm({ "w" => grad }, 2.0)

    assert_in_delta 5.0, total_norm.item, 1e-4
    assert_nested_close [1.2, 1.6], clipped.fetch("w").to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal expected.length, actual.length
    expected.zip(actual).each { |e, a| assert_in_delta e, a, atol }
  end
end
