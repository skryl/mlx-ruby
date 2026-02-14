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

    tree_flatten_calls = 0
    trace = TracePoint.new(:call) do |tp|
      next unless tp.method_id == :tree_flatten
      next unless tp.defined_class.equal?(MLX::Utils.singleton_class)

      tree_flatten_calls += 1
    end
    value = nil
    grads = nil
    trace.enable do
      value, grads = wrapped.call(MLX::Core.array([1.0, 2.0], MLX::Core.float32))
    end

    assert_in_delta 3.0, value.item, 1e-5
    assert_equal({}, grads)
    assert_equal 0, tree_flatten_calls,
      "value_and_grad no-param fast path should not call MLX::Utils.tree_flatten"
  end
end
