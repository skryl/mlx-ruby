# frozen_string_literal: true

require_relative "../test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class DslDataPipelineUnitTest < Minitest::Test
  def test_pipeline_supports_chainable_map_filter_batch_take
    pipeline = MLX::DSL::Data
      .from([1, 2, 3, 4, 5])
      .map { |x| x * 2 }
      .filter { |x| x > 4 }
      .batch(2)
      .take(1)

    assert_equal [[6, 8]], pipeline.to_a
  end

  def test_pipeline_repeat_with_fixed_count
    pipeline = MLX::DSL::Data.from([:a, :b]).repeat(2)
    assert_equal [:a, :b, :a, :b], pipeline.to_a
  end

  def test_pipeline_map_is_lazy_until_iteration
    seen = []
    pipeline = MLX::DSL::Data.from([1, 2, 3]).map do |x|
      seen << x
      x * 10
    end

    assert_equal [], seen
    assert_equal [10, 20], pipeline.take(2).to_a
    assert_equal [1, 2], seen
  end

  def test_pipeline_map_supports_index_aware_callable
    pipeline = MLX::DSL::Data.from(%w[a b c]).map do |item, index|
      "#{index}:#{item}"
    end

    assert_equal ["0:a", "1:b", "2:c"], pipeline.to_a
  end

  def test_pipeline_filter_supports_keyword_index_callable
    pipeline = MLX::DSL::Data.from([10, 20, 30, 40]).filter do |item, index:|
      item if index.odd?
    end

    assert_equal [20, 40], pipeline.to_a
  end

  def test_pipeline_shuffle_is_seeded_and_repeatable
    left = MLX::DSL::Data.from((1..8).to_a).shuffle(seed: 123).to_a
    right = MLX::DSL::Data.from((1..8).to_a).shuffle(seed: 123).to_a
    other = MLX::DSL::Data.from((1..8).to_a).shuffle(seed: 456).to_a

    assert_equal left, right
    refute_equal left, other
    assert_equal (1..8).to_a.sort, left.sort
  end

  def test_pipeline_prefetch_preserves_item_order
    pipeline = MLX::DSL::Data.from([1, 2, 3, 4]).prefetch(2).map { |x| x * 10 }
    assert_equal [10, 20, 30, 40], pipeline.to_a
  end

  def test_pipeline_prefetch_validates_positive_size
    error = assert_raises(ArgumentError) do
      MLX::DSL::Data.from([1]).prefetch(0).to_a
    end
    assert_match(/prefetch/i, error.message)
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
