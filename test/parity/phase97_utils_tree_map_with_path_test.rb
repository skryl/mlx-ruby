# frozen_string_literal: true

require_relative "test_helper"

class Phase97UtilsTreeMapWithPathTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_tree_map_with_path_reports_expected_paths
    tree = { "model" => [{ "w" => 1, "b" => 2 }, { "w" => 3, "b" => 4 }] }
    seen = []
    out = MLX::Utils.tree_map_with_path(
      lambda do |path, value|
        seen << path
        value * 10
      end,
      tree
    )

    assert_equal({ "model" => [{ "w" => 10, "b" => 20 }, { "w" => 30, "b" => 40 }] }, out)
    assert_equal %w[model.0.w model.0.b model.1.w model.1.b], seen
  end
end
