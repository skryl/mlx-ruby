# frozen_string_literal: true

require_relative "test_helper"

class Phase96UtilsTreeMapTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_tree_map_basic_and_multi_tree
    tree = { "a" => [1, 2], "b" => { "c" => 3 } }
    out = MLX::Utils.tree_map(->(x) { x * 2 }, tree)
    assert_equal({ "a" => [2, 4], "b" => { "c" => 6 } }, out)

    t2 = { "a" => [10, 20], "b" => { "c" => 30 } }
    summed = MLX::Utils.tree_map(->(x, y) { x + y }, tree, t2)
    assert_equal({ "a" => [11, 22], "b" => { "c" => 33 } }, summed)
  end

  def test_tree_map_is_leaf_override
    tree = { "a" => [1, 2], "b" => [3, 4] }
    out = MLX::Utils.tree_map(->(x) { x.length }, tree, is_leaf: ->(x) { x.is_a?(Array) })
    assert_equal({ "a" => 2, "b" => 2 }, out)
  end
end
