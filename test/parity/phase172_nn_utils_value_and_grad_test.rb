# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase172NnUtilsValueAndGradTest < Minitest::Test
  class LinearLike < MLX::NN::Module
    def initialize
      super()
      self.weight = MLX::Core.array([2.0], MLX::Core.float32)
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_value_and_grad_tracks_model_trainable_parameters
    model = LinearLike.new
    fn = lambda do |x|
      MLX::Core.sum(MLX::Core.multiply(model.weight, x))
    end

    wrapped = MLX::NN.value_and_grad(model, fn)
    value, grads = wrapped.call(MLX::Core.array([3.0], MLX::Core.float32))

    assert_in_delta 6.0, value.item, 1e-5
    assert grads.is_a?(Hash)
    assert grads.key?("weight")
    assert_nested_close [3.0], grads.fetch("weight").to_a, 1e-4
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal expected.length, actual.length
    expected.zip(actual).each { |e, a| assert_in_delta e, a, atol }
  end
end
