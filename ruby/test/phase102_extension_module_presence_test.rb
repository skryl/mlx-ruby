# frozen_string_literal: true

require_relative "test_helper"

class Phase102ExtensionModulePresenceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_extension_module_is_present
    assert defined?(MLX::Extension), "expected MLX::Extension to be defined"
    assert_kind_of Module, MLX::Extension
  end
end
