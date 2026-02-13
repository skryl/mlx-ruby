# frozen_string_literal: true

require_relative "test_helper"

class Phase36NanToNumTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_nan_to_num_with_explicit_replacements
    x = MLX::Core.array([MLX::Core.nan, MLX::Core.inf, -MLX::Core.inf, 1.5], MLX::Core.float32)

    out = MLX::Core.nan_to_num(x, 9.0, 8.0, -7.0)
    assert_nested_close [9.0, 8.0, -7.0, 1.5], out.to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal expected.length, actual.length
    expected.zip(actual).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end
end
