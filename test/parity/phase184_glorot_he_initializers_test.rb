# frozen_string_literal: true

require_relative "test_helper"

class Phase184GlorotHeInitializersTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    MLX::Core.random_seed(123)
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_glorot_and_he_uniform_bounds
    ref = MLX::Core.zeros([4, 2], MLX::Core.float32)

    glorot_u = MLX::NN.glorot_uniform(dtype: MLX::Core.float32)
    he_u_in = MLX::NN.he_uniform(dtype: MLX::Core.float32)
    he_u_out = MLX::NN.he_uniform(dtype: MLX::Core.float32)

    out_glorot = glorot_u.call(ref)
    out_he_in = he_u_in.call(ref, mode: "fan_in", gain: 1.0)
    out_he_out = he_u_out.call(ref, mode: "fan_out", gain: 1.0)

    glorot_limit = Math.sqrt(6.0 / (2 + 4))
    he_in_limit = Math.sqrt(3.0 / 2.0)
    he_out_limit = Math.sqrt(3.0 / 4.0)

    flatten(out_glorot.to_a).each { |v| assert_operator v, :<=, glorot_limit + 1e-6; assert_operator v, :>=, -glorot_limit - 1e-6 }
    flatten(out_he_in.to_a).each { |v| assert_operator v, :<=, he_in_limit + 1e-6; assert_operator v, :>=, -he_in_limit - 1e-6 }
    flatten(out_he_out.to_a).each { |v| assert_operator v, :<=, he_out_limit + 1e-6; assert_operator v, :>=, -he_out_limit - 1e-6 }
  end

  def test_glorot_and_he_normal_scale
    ref1 = MLX::Core.zeros([100, 100], MLX::Core.float32)
    ref2 = MLX::Core.zeros([100, 200], MLX::Core.float32)

    g = MLX::NN.glorot_normal(dtype: MLX::Core.float32).call(ref1)
    h = MLX::NN.he_normal(dtype: MLX::Core.float32).call(ref2, mode: "fan_in", gain: 1.0)

    g_std = std(flatten(g.to_a))
    h_std = std(flatten(h.to_a))

    assert_in_delta Math.sqrt(2.0 / 200.0), g_std, 0.02
    assert_in_delta 1.0 / Math.sqrt(200.0), h_std, 0.02
  end

  private

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |x| flatten(x) }
  end

  def std(values)
    mean = values.sum / values.length.to_f
    variance = values.sum { |v| (v - mean)**2 } / values.length.to_f
    Math.sqrt(variance)
  end
end
