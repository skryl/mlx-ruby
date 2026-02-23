# frozen_string_literal: true

require_relative "../test_helper"
require_relative "../../examples/benchmark/benchmark_mlx_examples"

class Phase325ExamplesAdapterBooleanDiffParityTest < Minitest::Test
  def setup
    @adapter = ExamplesModelsBenchmarkAdapter.new(
      repo_root: RUBY_ROOT,
      submodule_root: File.join(RUBY_ROOT, "mlx-ruby-examples"),
      device: :cpu,
      runs: 1,
      warmup: 0,
      timeout: 1,
      mode: :dsl
    )
  end

  def test_max_abs_diff_supports_boolean_scalars
    assert_equal 0.0, diff(true, true)
    assert_equal 0.0, diff(false, false)
    assert_equal 1.0, diff(true, false)
    assert_equal 1.0, diff(false, true)
  end

  def test_max_abs_diff_supports_nested_boolean_structures
    expected = [
      { "flag" => true, "values" => [1.0, false] },
      { "score" => 0.5 }
    ]
    actual = [
      { "flag" => false, "values" => [1.0, true] },
      { "score" => 0.75 }
    ]

    assert_equal 1.0, diff(expected, actual)
  end

  def test_max_abs_diff_supports_nil_and_boolean_paths
    assert_equal 1.0, diff(nil, true)
    assert_equal 0.0, diff(false, nil)
  end

  private

  def diff(expected, actual)
    @adapter.send(:max_abs_diff, expected, actual)
  end
end
