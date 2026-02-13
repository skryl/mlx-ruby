# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase191PositionalEncodingParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_rope_matches_core_rope
    x = MLX::Core.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], MLX::Core.float32)
    layer = MLX::NN::RoPE.new(4, traditional: true, base: 10_000.0, scale: 1.0)

    expected = MLX::Core.rope(x, 4, true, 10_000.0, 1.0, 0)
    actual = layer.call(x)
    assert_nested_close expected.to_a, actual.to_a
  end

  def test_sinusoidal_positional_encoding_shape_and_order
    x = MLX::Core.array([0.0, 1.0, 2.0], MLX::Core.float32)

    sin_first = MLX::NN::SinusoidalPositionalEncoding.new(4, cos_first: false)
    cos_first = MLX::NN::SinusoidalPositionalEncoding.new(4, cos_first: true)

    y_sin = sin_first.call(x)
    y_cos = cos_first.call(x)

    assert_equal [3, 4], y_sin.shape
    assert_equal [3, 4], y_cos.shape

    scale = Math.sqrt(2.0 / 4.0)
    assert_in_delta 0.0, y_sin.to_a[0][0], 1e-6
    assert_in_delta scale, y_sin.to_a[0][2], 1e-6
    assert_in_delta scale, y_cos.to_a[0][0], 1e-6
    assert_in_delta 0.0, y_cos.to_a[0][2], 1e-6
  end

  def test_alibi_slope_matrix_and_call
    slopes = MLX::NN::ALiBi.create_alibi_slope(num_heads: 4, dtype: MLX::Core.float32)
    assert_equal [4, 1, 1], slopes.shape
    slope_values = slopes.to_a.map { |entry| entry[0][0] }
    assert_operator slope_values[0], :>, slope_values[1]

    matrix = MLX::NN::ALiBi.create_alibi_matrix(
      q_sequence_length: 4,
      k_sequence_length: 4,
      num_heads: 2,
      offset: 0,
      dtype: MLX::Core.float32
    )
    assert_equal [1, 2, 4, 4], matrix.shape
    matrix.to_a[0][0].each_with_index do |row, i|
      assert_in_delta 0.0, row[i], 1e-6
      row.each { |v| assert_operator v, :<=, 0.0 }
    end

    attention = MLX::Core.zeros([1, 2, 4, 4], MLX::Core.float32)
    mask = MLX::Core.full([1, 2, 4, 4], -0.25, MLX::Core.float32)
    out = MLX::NN::ALiBi.new.call(attention, offset: 0, mask: mask)

    expected = MLX::Core.add(matrix, mask)
    assert_nested_close expected.to_a, out.to_a
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
