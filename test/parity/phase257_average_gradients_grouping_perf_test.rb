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

  def with_stubbed_all_sum
    singleton = class << MLX::Core
      self
    end
    backup = :__phase257_original_all_sum
    visibility = singleton_method_visibility(singleton, :all_sum)

    remove_singleton_method(singleton, backup)
    if visibility
      singleton.send(:alias_method, backup, :all_sum)
      remove_singleton_method(singleton, :all_sum)
    end

    calls = [0]
    singleton.define_method(:all_sum) do |x, _stream = nil|
      calls[0] += 1
      x
    end

    yield
    calls[0]
  ensure
    remove_singleton_method(singleton, :all_sum) if defined?(singleton)

    if defined?(visibility) && visibility
      singleton.send(:alias_method, :all_sum, backup)
      singleton.send(:remove_method, backup)
      singleton.send(:private, :all_sum) if visibility == :private
      singleton.send(:protected, :all_sum) if visibility == :protected
    end
  end

  def singleton_method_visibility(singleton, name)
    return :private if singleton.private_instance_methods(false).include?(name)
    return :protected if singleton.protected_instance_methods(false).include?(name)
    return :public if singleton.instance_methods(false).include?(name)

    nil
  end

  def remove_singleton_method(singleton, name)
    return unless singleton.private_instance_methods(false).include?(name) ||
      singleton.protected_instance_methods(false).include?(name) ||
      singleton.instance_methods(false).include?(name)

    singleton.send(:remove_method, name)
  end

  def test_average_gradients_groups_small_arrays_into_single_all_sum
    grads = {
      "a" => MLX::Core.array([1.0, 2.0], MLX::Core.float32),
      "b" => MLX::Core.array([3.0, 4.0], MLX::Core.float32),
      "c" => MLX::Core.array([5.0, 6.0], MLX::Core.float32)
    }

    calls = with_stubbed_all_sum do
      averaged = MLX::NN.average_gradients(
        grads,
        FakeGroup.new(4),
        all_reduce_size: 1024 * 1024
      )

      assert_equal [0.25, 0.5], averaged.fetch("a").to_a
      assert_equal [0.75, 1.0], averaged.fetch("b").to_a
      assert_equal [1.25, 1.5], averaged.fetch("c").to_a
    end

    assert_equal 1, calls
  end

  def test_average_gradients_with_non_positive_group_size_avoids_grouping
    grads = {
      "a" => MLX::Core.array([1.0, 2.0], MLX::Core.float32),
      "b" => MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    }

    calls = with_stubbed_all_sum do
      _averaged = MLX::NN.average_gradients(
        grads,
        FakeGroup.new(2),
        all_reduce_size: 0
      )
    end

    assert_equal 2, calls
  end
end
