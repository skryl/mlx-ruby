# frozen_string_literal: true

require_relative "test_helper"

class Phase165MuonFunctionalParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_muon_1d_default_nesterov_update
    opt = MLX::Optimizers::Muon.new(learning_rate: 0.1)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out = opt.apply_gradients(grads, params)

    assert_nested_close [9.979525], out.fetch("w").to_a, 1e-6
    assert_nested_close [0.105], opt.state.fetch("w").fetch("v").to_a, 1e-6
  end

  def test_muon_1d_without_nesterov
    opt = MLX::Optimizers::Muon.new(learning_rate: 0.1, nesterov: false)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out = opt.apply_gradients(grads, params)

    assert_nested_close [9.9895], out.fetch("w").to_a, 1e-6
    assert_nested_close [0.105], opt.state.fetch("w").fetch("v").to_a, 1e-6
  end

  def test_muon_2d_newton_schulz_path
    opt = MLX::Optimizers::Muon.new(
      learning_rate: 0.1,
      momentum: 0.0,
      weight_decay: 0.0,
      nesterov: false,
      ns_steps: 0
    )

    params = {
      "w" => MLX::Core.array([[0.0, 0.0], [0.0, 0.0]], MLX::Core.float32)
    }
    grads = {
      "w" => MLX::Core.array([[3.0, 0.0], [0.0, 4.0]], MLX::Core.float32)
    }

    out = opt.apply_gradients(grads, params)

    expected = [[-0.06, 0.0], [0.0, -0.08]]
    assert_nested_close expected, out.fetch("w").to_a, 1e-5
    assert_equal [2, 2], opt.state.fetch("w").fetch("v").shape
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
