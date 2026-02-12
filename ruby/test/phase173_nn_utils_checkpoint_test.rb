# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase173NnUtilsCheckpointTest < Minitest::Test
  class DoubleModule < MLX::NN::Module
    def call(x)
      MLX::Core.multiply(x, 2.0)
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_checkpoint_with_explicit_fn
    mod = DoubleModule.new
    fn = lambda do |x|
      MLX::Core.add(mod.call(x), 1.0)
    end

    checkpointed = MLX::NN.checkpoint(mod, fn)
    out = checkpointed.call(MLX::Core.array([3.0], MLX::Core.float32))

    assert_nested_close [7.0], out.to_a
  end

  def test_checkpoint_defaults_to_module_call
    mod = DoubleModule.new

    checkpointed = MLX::NN.checkpoint(mod)
    out = checkpointed.call(MLX::Core.array([4.0], MLX::Core.float32))

    assert_nested_close [8.0], out.to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal expected.length, actual.length
    expected.zip(actual).each { |e, a| assert_in_delta e, a, atol }
  end
end
