# frozen_string_literal: true

require_relative "test_helper"

class Phase186InitializerValidationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_he_mode_validation
    ref = MLX::Core.zeros([4, 2], MLX::Core.float32)

    assert_raises(ArgumentError) do
      MLX::NN.he_normal.call(ref, mode: "invalid")
    end

    assert_raises(ArgumentError) do
      MLX::NN.he_uniform.call(ref, mode: "invalid")
    end
  end

  def test_sparse_and_orthogonal_ndim_validation
    ref3d = MLX::Core.zeros([2, 2, 2], MLX::Core.float32)

    assert_raises(ArgumentError) do
      MLX::NN.sparse(sparsity: 0.5).call(ref3d)
    end

    assert_raises(ArgumentError) do
      MLX::NN.orthogonal.call(ref3d)
    end
  end
end
