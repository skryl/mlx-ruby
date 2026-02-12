# frozen_string_literal: true

require_relative "test_helper"

class Phase131PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_131_contract
    fn = MLX::NN.value_and_grad(MLX::NN::Module.new, ->(x) { x })
    out = fn.call(MLX::Core.array([1.0], MLX::Core.float32))
    assert_kind_of Array, out
  end
end
