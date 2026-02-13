# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase169ModuleTraversalTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def build_tree
    root = MLX::NN::Module.new

    root.a = MLX::NN::Module.new
    root.a.w = MLX::Core.array([1.0], MLX::Core.float32)

    root.b = { "x" => MLX::NN::Module.new }
    root.b["x"].v = MLX::Core.array([2.0], MLX::Core.float32)

    root.c = [MLX::NN::Module.new, "noop"]
    root.c[0].q = MLX::Core.array([3.0], MLX::Core.float32)

    root
  end

  def test_children_preserve_direct_module_structure
    root = build_tree

    children = root.children

    assert children.key?("a")
    assert children.key?("b")
    assert children.key?("c")
    assert_kind_of MLX::NN::Module, children["a"]
    assert_kind_of MLX::NN::Module, children["b"]["x"]
    assert_kind_of MLX::NN::Module, children["c"][0]
  end

  def test_modules_and_named_modules_traverse_all_modules
    root = build_tree

    mods = root.modules
    names = root.named_modules.map(&:first).sort

    assert_equal 4, mods.length
    assert_equal ["", "a", "b.x", "c.0"], names
  end

  def test_leaf_modules_only_returns_leaf_submodules
    root = build_tree

    leaves = root.leaf_modules

    assert leaves.key?("a")
    assert leaves.key?("b")
    assert leaves.key?("c")
    assert_kind_of MLX::NN::Module, leaves["a"]
    assert_kind_of MLX::NN::Module, leaves["b"]["x"]
    assert_kind_of MLX::NN::Module, leaves["c"][0]
  end
end
