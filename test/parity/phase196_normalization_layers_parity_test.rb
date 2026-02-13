# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase196NormalizationLayersParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_layer_norm_and_rms_norm_match_core_ops
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)

    ln = MLX::NN::LayerNorm.new(2, eps: 1e-5, affine: true, bias: true)
    ln.weight = MLX::Core.array([1.5, 0.5], MLX::Core.float32)
    ln.bias = MLX::Core.array([0.1, -0.2], MLX::Core.float32)
    expected_ln = MLX::Core.layer_norm(x, ln.weight, ln.bias, 1e-5)
    assert_nested_close expected_ln.to_a, ln.call(x).to_a

    rms = MLX::NN::RMSNorm.new(2, eps: 1e-5)
    rms.weight = MLX::Core.array([2.0, 3.0], MLX::Core.float32)
    expected_rms = MLX::Core.rms_norm(x, rms.weight, 1e-5)
    assert_nested_close expected_rms.to_a, rms.call(x).to_a
  end

  def test_instance_norm_normalizes_per_instance
    x = MLX::Core.array(
      [
        [[[1.0, 3.0], [2.0, 4.0]], [[3.0, 5.0], [4.0, 6.0]]],
        [[[2.0, 4.0], [3.0, 5.0]], [[4.0, 6.0], [5.0, 7.0]]]
      ],
      MLX::Core.float32
    )

    layer = MLX::NN::InstanceNorm.new(2, affine: true)
    layer.weight = MLX::Core.array([1.0, 1.0], MLX::Core.float32)
    layer.bias = MLX::Core.array([0.0, 0.0], MLX::Core.float32)
    y = layer.call(x).to_a

    2.times do |b|
      2.times do |c|
        values = [y[b][0][0][c], y[b][0][1][c], y[b][1][0][c], y[b][1][1][c]]
        mean = values.sum / values.length.to_f
        assert_in_delta 0.0, mean, 1e-4
      end
    end
  end

  def test_group_norm_forward_paths
    x = MLX::Core.array(
      [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
      MLX::Core.float32
    )

    gn = MLX::NN::GroupNorm.new(2, 4, affine: true, pytorch_compatible: false)
    out = gn.call(x)
    assert_equal [1, 2, 2, 4], out.shape

    gn_pt = MLX::NN::GroupNorm.new(2, 4, affine: true, pytorch_compatible: true)
    out_pt = gn_pt.call(x)
    assert_equal [1, 2, 2, 4], out_pt.shape
  end

  def test_batch_norm_training_updates_running_stats_and_eval_uses_them
    bn = MLX::NN::BatchNorm.new(2, momentum: 0.5, affine: true, track_running_stats: true)
    x = MLX::Core.array([[1.0, 2.0], [5.0, 8.0]], MLX::Core.float32)

    _ = bn.call(x)
    refute_nested_equal [0.0, 0.0], bn.running_mean.to_a
    refute_nested_equal [1.0, 1.0], bn.running_var.to_a

    bn.eval
    x_eval = MLX::Core.array([[10.0, 20.0], [30.0, 40.0]], MLX::Core.float32)
    out = bn.call(x_eval)

    centered = MLX::Core.subtract(x_eval, bn.running_mean)
    expected = MLX::Core.multiply(centered, MLX::Core.rsqrt(MLX::Core.add(bn.running_var, bn.eps)))
    expected = MLX::Core.add(MLX::Core.multiply(expected, bn.weight), bn.bias)
    assert_nested_close expected.to_a, out.to_a
  end

  def test_batch_norm_input_rank_validation
    bn = MLX::NN::BatchNorm.new(2)
    bad = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    assert_raises(ArgumentError) { bn.call(bad) }
  end

  private

  def refute_nested_equal(expected, actual)
    refute_equal expected.flatten, actual.flatten
  end

  def assert_nested_close(expected, actual, atol = 1e-4)
    assert_equal shape_signature(expected), shape_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |item| flatten(item) }
  end

  def shape_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |item| shape_signature(item) })]
  end
end
