# frozen_string_literal: true

require_relative "test_helper"

class Phase166AdvancedOptimizerIntegrationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_adafactor_and_muon_update_nested_trees
    params = {
      "layer1" => {
        "w" => MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32),
        "b" => MLX::Core.array([0.5, -0.5], MLX::Core.float32)
      },
      "layer2" => [MLX::Core.array([1.0, 1.0], MLX::Core.float32)]
    }
    grads = {
      "layer1" => {
        "w" => MLX::Core.array([[0.1, 0.1], [0.1, 0.1]], MLX::Core.float32),
        "b" => MLX::Core.array([0.2, -0.2], MLX::Core.float32)
      },
      "layer2" => [MLX::Core.array([0.3, 0.3], MLX::Core.float32)]
    }

    adafactor = MLX::Optimizers::Adafactor.new(learning_rate: 0.05, relative_step: false, scale_parameter: false)
    out_a = adafactor.apply_gradients(grads, params)

    muon = MLX::Optimizers::Muon.new(learning_rate: 0.01, momentum: 0.9, weight_decay: 0.0, ns_steps: 1)
    out_m = muon.apply_gradients(grads, params)

    assert_equal 1, adafactor.step
    assert_equal 1, muon.step

    refute_equal params["layer1"]["w"].to_a, out_a["layer1"]["w"].to_a
    refute_equal params["layer1"]["w"].to_a, out_m["layer1"]["w"].to_a
    refute_equal params["layer1"]["b"].to_a, out_a["layer1"]["b"].to_a
    refute_equal params["layer1"]["b"].to_a, out_m["layer1"]["b"].to_a
  end

  def test_multi_optimizer_routes_advanced_optimizers_by_path
    matrix_opt = MLX::Optimizers::Muon.new(learning_rate: 0.02, momentum: 0.9, weight_decay: 0.0)
    vector_opt = MLX::Optimizers::Adafactor.new(learning_rate: 0.03, relative_step: false, scale_parameter: false)

    filter = lambda do |path, _grad|
      path.end_with?(".w")
    end

    multi = MLX::Optimizers::MultiOptimizer.new([matrix_opt, vector_opt], filters: [filter])

    params = {
      "blk" => {
        "w" => MLX::Core.array([[2.0, 0.0], [0.0, 2.0]], MLX::Core.float32),
        "b" => MLX::Core.array([0.1, -0.1], MLX::Core.float32)
      }
    }
    grads = {
      "blk" => {
        "w" => MLX::Core.array([[0.5, 0.5], [0.5, 0.5]], MLX::Core.float32),
        "b" => MLX::Core.array([0.4, -0.4], MLX::Core.float32)
      }
    }

    out = multi.apply_gradients(grads, params)

    assert_equal 1, matrix_opt.step
    assert_equal 1, vector_opt.step
    refute_equal params["blk"]["w"].to_a, out["blk"]["w"].to_a
    refute_equal params["blk"]["b"].to_a, out["blk"]["b"].to_a
  end
end
