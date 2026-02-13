# frozen_string_literal: true

require_relative "test_helper"

class Phase80ModulePresenceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_top_level_modules_are_present
    assert defined?(MLX::Core)
    assert defined?(MLX::Utils), "MLX::Utils should exist"
    assert defined?(MLX::NN), "MLX::NN should exist"
    assert defined?(MLX::Optimizers), "MLX::Optimizers should exist"
  end

  def test_utils_tree_helpers_surface_exists
    assert_respond_to MLX::Utils, :tree_map
    assert_respond_to MLX::Utils, :tree_map_with_path
    assert_respond_to MLX::Utils, :tree_flatten
    assert_respond_to MLX::Utils, :tree_unflatten
    assert_respond_to MLX::Utils, :tree_reduce
    assert_respond_to MLX::Utils, :tree_merge
  end
end
