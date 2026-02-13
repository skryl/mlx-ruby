# frozen_string_literal: true

require_relative "test_helper"
$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase262NnValueAndGradTreeScanPerfTest < Minitest::Test
  class NoParamModel < MLX::NN::Module
    def initialize
      super()
      self.name = "no-params"
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    nil
  end

  def test_value_and_grad_no_param_path_does_not_require_tree_flatten
    model = NoParamModel.new
    fn = lambda do |x|
      MLX::Core.sum(x)
    end

    wrapped = MLX::NN.value_and_grad(model, fn)

    original_tree_flatten = MLX::Utils.method(:tree_flatten)
    MLX::Utils.define_singleton_method(:tree_flatten) do |*|
      raise "value_and_grad no-param fast path should not call tree_flatten"
    end

    value, grads = wrapped.call(MLX::Core.array([1.0, 2.0], MLX::Core.float32))

    assert_in_delta 3.0, value.item, 1e-5
    assert_equal({}, grads)
  ensure
    MLX::Utils.define_singleton_method(:tree_flatten, original_tree_flatten) unless original_tree_flatten.nil?
  end
end
