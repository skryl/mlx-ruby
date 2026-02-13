# frozen_string_literal: true

require_relative "test_helper"

class Phase275MultiOptimizerSplitCachePerfTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_multi_optimizer_caches_filter_assignments_for_stable_paths
    calls = 0
    filter = lambda do |path, _grad|
      calls += 1
      path.start_with?("left")
    end

    optim = MLX::Optimizers::MultiOptimizer.new(
      [
        MLX::Optimizers::SGD.new(learning_rate: 0.1),
        MLX::Optimizers::SGD.new(learning_rate: 0.1)
      ],
      filters: [filter]
    )

    grads = {
      "left" => MLX::Core.array([1.0], MLX::Core.float32),
      "right" => MLX::Core.array([1.0], MLX::Core.float32)
    }
    params = {
      "left" => MLX::Core.array([2.0], MLX::Core.float32),
      "right" => MLX::Core.array([3.0], MLX::Core.float32)
    }

    optim.apply_gradients(grads, params)
    optim.apply_gradients(grads, params)

    assert_operator calls, :<=, 2, "expected path-based split filter calls to be cached after first pass"
  end
end
