# frozen_string_literal: true

require_relative "test_helper"

class Phase109PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_109_contract
    ext = MLX::Extension::CMakeExtension.new("mlx_ext", "src")
    assert_equal "mlx_ext", ext.name
    assert_equal true, MLX::Extension::CMakeBuild.new.run
  end
end
