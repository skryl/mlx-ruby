# frozen_string_literal: true

require_relative "test_helper"

class Phase31ConstructLikeTriTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_full_and_like_constructors
    x = MLX::Core.array([[1, 2], [3, 4]], MLX::Core.int32)

    assert_equal [[0, 0], [0, 0]], MLX::Core.zeros_like(x).to_a
    assert_equal [[1, 1], [1, 1]], MLX::Core.ones_like(x).to_a
    assert_equal [[7, 7, 7], [7, 7, 7]], MLX::Core.full([2, 3], 7, MLX::Core.int32).to_a
  end

  def test_eye_identity_tri_tril_triu
    assert_equal [[1, 0, 0], [0, 1, 0], [0, 0, 1]], MLX::Core.eye(3, 3, 0, MLX::Core.int32).to_a
    assert_equal [[1, 0, 0], [0, 1, 0], [0, 0, 1]], MLX::Core.identity(3, MLX::Core.int32).to_a
    assert_equal [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]], MLX::Core.tri(3, 4, 0, MLX::Core.int32).to_a

    m = MLX::Core.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], MLX::Core.int32)
    assert_equal [[1, 0, 0], [4, 5, 0], [7, 8, 9]], MLX::Core.tril(m).to_a
    assert_equal [[0, 2, 3], [0, 0, 6], [0, 0, 0]], MLX::Core.triu(m, 1).to_a
  end
end
