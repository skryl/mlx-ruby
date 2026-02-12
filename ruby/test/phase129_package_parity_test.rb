# frozen_string_literal: true

require_relative "test_helper"

class Phase129PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_129_contract
    mod = MLX::NN::Module.new
    assert_same mod, mod.update({"w" => 1}, strict: false)
    assert_same mod, mod.apply { |_n, _m| }
  end
end
