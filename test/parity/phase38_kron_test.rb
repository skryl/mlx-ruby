# frozen_string_literal: true

require_relative "test_helper"

class Phase38KronTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_kron_product
    a = MLX::Core.array([[1, 2], [3, 4]], MLX::Core.int32)
    b = MLX::Core.array([[0, 5], [6, 7]], MLX::Core.int32)

    out = MLX::Core.kron(a, b)
    assert_equal [4, 4], out.shape
    assert_equal [[0, 5, 0, 10], [6, 7, 12, 14], [0, 15, 0, 20], [18, 21, 24, 28]], out.to_a
  end
end
