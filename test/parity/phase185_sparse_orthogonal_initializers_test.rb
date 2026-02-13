# frozen_string_literal: true

require_relative "test_helper"

class Phase185SparseOrthogonalInitializersTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    MLX::Core.random_seed(123)
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_sparse_initializer_sets_expected_zeros_per_row
    ref = MLX::Core.zeros([4, 6], MLX::Core.float32)
    init = MLX::NN.sparse(sparsity: 0.5, mean: 0.0, std: 1.0, dtype: MLX::Core.float32)

    out = init.call(ref).to_a
    values = out.flatten
    zeros_ratio = values.count { |v| v.abs < 1e-12 }.to_f / values.length
    assert_in_delta 0.5, zeros_ratio, 0.25
  end

  def test_orthogonal_initializer_returns_orthogonal_matrix_scaled_by_gain
    with_cpu_default_device do
      ref = MLX::Core.zeros([3, 3], MLX::Core.float32)
      gain = 1.5
      init = MLX::NN.orthogonal(gain: gain, dtype: MLX::Core.float32)

      q = init.call(ref)
      qtq = MLX::Core.matmul(q.T, q).to_a

      3.times do |i|
        3.times do |j|
          if i == j
            assert_in_delta gain * gain, qtq[i][j], 1e-2
          else
            assert_in_delta 0.0, qtq[i][j], 1e-2
          end
        end
      end
    end
  end

  private

  def with_cpu_default_device
    previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    yield
  ensure
    MLX::Core.set_default_device(previous_device) if previous_device
  end
end
