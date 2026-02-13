# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase197RecurrentLayersParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_rnn_forward_with_custom_nonlinearity
    assert_raises(ArgumentError) { MLX::NN::RNN.new(2, 2, nonlinearity: 123) }

    identity = ->(z) { z }
    rnn = MLX::NN::RNN.new(2, 2, bias: false, nonlinearity: identity)
    rnn.Wxh = MLX::Core.array([[1.0, 0.0], [0.0, 1.0]], MLX::Core.float32)
    rnn.Whh = MLX::Core.array([[1.0, 0.0], [0.0, 1.0]], MLX::Core.float32)

    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    out = rnn.call(x)

    assert_nested_close [[1.0, 2.0], [4.0, 6.0]], out.to_a
  end

  def test_gru_forward_zero_weights
    gru = MLX::NN::GRU.new(2, 2, bias: true)
    gru.Wx = MLX::Core.zeros([6, 2], MLX::Core.float32)
    gru.Wh = MLX::Core.zeros([6, 2], MLX::Core.float32)
    gru.b = MLX::Core.zeros([6], MLX::Core.float32)
    gru.bhn = MLX::Core.zeros([2], MLX::Core.float32)

    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    out = gru.call(x)

    assert_equal [2, 2], out.shape
    assert_nested_close [[0.0, 0.0], [0.0, 0.0]], out.to_a
  end

  def test_lstm_forward_zero_weights
    lstm = MLX::NN::LSTM.new(2, 2, bias: true)
    lstm.Wx = MLX::Core.zeros([8, 2], MLX::Core.float32)
    lstm.Wh = MLX::Core.zeros([8, 2], MLX::Core.float32)
    lstm.bias = MLX::Core.zeros([8], MLX::Core.float32)

    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    hidden, cell = lstm.call(x)

    assert_equal [2, 2], hidden.shape
    assert_equal [2, 2], cell.shape
    assert_nested_close [[0.0, 0.0], [0.0, 0.0]], hidden.to_a
    assert_nested_close [[0.0, 0.0], [0.0, 0.0]], cell.to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
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
