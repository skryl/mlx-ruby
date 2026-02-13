# frozen_string_literal: true

require_relative "test_helper"

class Phase257AverageGradientsGroupingPerfTest < Minitest::Test
  FakeGroup = Struct.new(:size)

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_average_gradients_groups_small_arrays_into_single_all_sum
    grads = {
      "a" => MLX::Core.array([1.0, 2.0], MLX::Core.float32),
      "b" => MLX::Core.array([3.0, 4.0], MLX::Core.float32),
      "c" => MLX::Core.array([5.0, 6.0], MLX::Core.float32)
    }

    calls = 0
    original_all_sum = MLX::Core.method(:all_sum)

    MLX::Core.define_singleton_method(:all_sum) do |x, _stream = nil|
      calls += 1
      x
    end

    averaged = MLX::NN.average_gradients(
      grads,
      FakeGroup.new(4),
      all_reduce_size: 1024 * 1024
    )

    assert_equal 1, calls
    assert_equal [0.25, 0.5], averaged.fetch("a").to_a
    assert_equal [0.75, 1.0], averaged.fetch("b").to_a
    assert_equal [1.25, 1.5], averaged.fetch("c").to_a
  ensure
    MLX::Core.define_singleton_method(:all_sum, original_all_sum)
  end

  def test_average_gradients_with_non_positive_group_size_avoids_grouping
    grads = {
      "a" => MLX::Core.array([1.0, 2.0], MLX::Core.float32),
      "b" => MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    }

    calls = 0
    original_all_sum = MLX::Core.method(:all_sum)

    MLX::Core.define_singleton_method(:all_sum) do |x, _stream = nil|
      calls += 1
      x
    end

    _averaged = MLX::NN.average_gradients(
      grads,
      FakeGroup.new(2),
      all_reduce_size: 0
    )

    assert_equal 2, calls
  ensure
    MLX::Core.define_singleton_method(:all_sum, original_all_sum)
  end
end
