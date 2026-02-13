# frozen_string_literal: true

require_relative "test_helper"

class Phase128PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_128_contract
    mod = MLX::NN::Module.new
    assert_equal({}, mod.children)
    named = mod.named_modules
    assert_kind_of Array, named
    assert_equal 1, named.length
    assert_equal "", named[0][0]
    assert_same mod, named[0][1]
  end
end
