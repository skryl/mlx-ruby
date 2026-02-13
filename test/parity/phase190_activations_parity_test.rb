# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase190ActivationsParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_relu_family_and_leaky_relu
    x = MLX::Core.array([-2.0, -0.5, 0.0, 0.5, 8.0], MLX::Core.float32)

    assert_nested_close [0.0, 0.0, 0.0, 0.5, 8.0], MLX::NN.relu(x).to_a
    assert_nested_close [0.0, 0.0, 0.0, 0.25, 64.0], MLX::NN.relu2(x).to_a
    assert_nested_close [0.0, 0.0, 0.0, 0.5, 6.0], MLX::NN.relu6(x).to_a
    assert_nested_close [-0.2, -0.05, 0.0, 0.5, 8.0], MLX::NN.leaky_relu(x, negative_slope: 0.1).to_a

    layer = MLX::NN::LeakyReLU.new(0.1)
    assert_nested_close [-0.2, -0.05, 0.0, 0.5, 8.0], layer.call(x).to_a
  end

  def test_softmax_log_softmax_and_softmin
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    assert_nested_close [0.26894143, 0.7310586], MLX::NN.softmax(x).to_a
    assert_nested_close [-1.3132616, -0.31326163], MLX::NN.log_softmax(x).to_a
    assert_nested_close [0.7310586, 0.26894143], MLX::NN.softmin(x).to_a

    assert_nested_close [0.26894143, 0.7310586], MLX::NN::Softmax.new.call(x).to_a
    assert_nested_close [-1.3132616, -0.31326163], MLX::NN::LogSoftmax.new.call(x).to_a
  end

  def test_shrink_threshold_and_tanh_family
    x = MLX::Core.array([-2.0, -0.25, 0.0, 0.25, 2.0], MLX::Core.float32)

    assert_nested_close [-1.5, 0.0, 0.0, 0.0, 1.5], MLX::NN.softshrink(x).to_a
    assert_nested_close [-2.0, 0.0, 0.0, 0.0, 2.0], MLX::NN.hard_shrink(x).to_a
    assert_nested_close [0, 0, 0, 0, 1], MLX::NN.step(x, threshold: 0.25).to_a
    assert_nested_close [-1.0, -0.25, 0.0, 0.25, 1.0], MLX::NN.hard_tanh(x).to_a
    assert_nested_close [-0.9640276, -0.24491866, 0.0, 0.24491866, 0.9640276], MLX::NN.tanh(x).to_a
  end

  def test_softplus_sigmoid_family
    x = MLX::Core.array([-1.0, 0.0, 1.0], MLX::Core.float32)

    assert_nested_close [0.3132617, 0.6931472, 1.3132616], MLX::NN.softplus(x).to_a
    assert_nested_close [-1.3132616, -0.6931472, -0.31326163], MLX::NN.log_sigmoid(x).to_a
    assert_nested_close [-0.26894143, 0.0, 0.7310586], MLX::NN.silu(x).to_a

    mish_expected = MLX::Core.multiply(x, MLX::Core.tanh(MLX::NN.softplus(x)))
    assert_nested_close mish_expected.to_a, MLX::NN.mish(x).to_a
  end

  def test_glu_prelu_and_gelu_module_behavior
    x_glu = MLX::Core.array([[1.0, 2.0, -1.0, -2.0]], MLX::Core.float32)
    glu = MLX::NN::GLU.new
    assert_nested_close [[0.26894143, 0.23840584]], glu.call(x_glu).to_a

    x_prelu = MLX::Core.array([-2.0, 2.0], MLX::Core.float32)
    alpha = MLX::Core.array([0.25], MLX::Core.float32)
    assert_nested_close [-0.5, 2.0], MLX::NN.prelu(x_prelu, alpha).to_a

    prelu = MLX::NN::PReLU.new(1, init: 0.1)
    assert_equal [1], prelu.weight.shape
    assert_nested_close [-0.2, 2.0], prelu.call(x_prelu).to_a

    assert_raises(ArgumentError) { MLX::NN::GELU.new("invalid") }
    gelu_precise = MLX::NN::GELU.new("precise")
    gelu_fast = MLX::NN::GELU.new("fast")
    out_precise = gelu_precise.call(MLX::Core.array([1.0], MLX::Core.float32)).to_a.first
    out_fast = gelu_fast.call(MLX::Core.array([1.0], MLX::Core.float32)).to_a.first
    refute_in_delta out_precise, out_fast, 1e-3
  end

  private

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
