# frozen_string_literal: true

require_relative "../support/test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class ModuleUpdateModulesRecursionTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_update_modules_recurses_when_current_value_is_module_and_new_value_is_hash
    root = build_tree
    replacement = build_leaf(9.0)

    root.update_modules(
      { "child" => { "inner" => replacement } },
      strict: true
    )

    assert_same replacement, root.child.inner
  end

  def test_update_modules_recurses_for_array_of_modules_when_new_values_are_hashes
    root = build_tree
    replacement = build_leaf(11.0)

    root.update_modules(
      { "items" => [{ "inner" => replacement }] },
      strict: true
    )

    assert_same replacement, root.items[0].inner
  end

  private

  def build_leaf(value)
    leaf = MLX::NN::Module.new
    leaf.weight = MLX::Core.array([value], MLX::Core.float32)
    leaf
  end

  def build_tree
    root = MLX::NN::Module.new
    root.child = MLX::NN::Module.new
    root.child.inner = build_leaf(1.0)

    root.items = [MLX::NN::Module.new]
    root.items[0].inner = build_leaf(2.0)
    root
  end
end
