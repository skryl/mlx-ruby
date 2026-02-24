# frozen_string_literal: true

require_relative "test_helper"

class Phase327ValueAndGradNativeCacheParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_value_and_grad_reuses_native_transform_for_same_structure
    original = MLX::Core.method(:native_value_and_grad)
    native_builds = 0

    install_singleton_override(:native_value_and_grad) do |fun, argnums|
      native_builds += 1
      original.call(fun, argnums)
    end

    fn = lambda do |x|
      MLX::Core.sum(MLX::Core.multiply(x, x))
    end
    wrapped = MLX::Core.value_and_grad(fn)

    3.times do
      value, grad = wrapped.call(MLX::Core.array([2.0, -3.0], MLX::Core.float32))
      MLX::Core.eval(value, grad)
      assert_in_delta 13.0, value.item, 1e-5
      assert_equal [4.0, -6.0], grad.to_a
    end

    assert_equal 1, native_builds
  ensure
    restore_singleton_override(:native_value_and_grad)
  end

  def test_grad_sees_updated_non_target_arguments_across_calls
    loss = lambda do |x, y:|
      MLX::Core.sum(MLX::Core.multiply(x, y))
    end
    grad_fn = MLX::Core.grad(loss, nil, ["y"])

    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    first = grad_fn.call(MLX::Core.array([1.0, 2.0], MLX::Core.float32), y: y)
    second = grad_fn.call(MLX::Core.array([10.0, 20.0], MLX::Core.float32), y: y)

    assert_nil first[0]
    assert_nil second[0]
    assert_equal [1.0, 2.0], first[1]["y"].to_a
    assert_equal [10.0, 20.0], second[1]["y"].to_a
  end

  def test_value_and_grad_reentrant_call_with_same_selection_key_is_safe
    reentered = false
    inner_observation = nil
    wrapped = nil

    install_singleton_override(:native_value_and_grad) do |lifted, _argnums|
      lambda do |*flat_vars|
        unless reentered
          reentered = true
          inner_value, inner_grad = wrapped.call(MLX::Core.array([5.0, -1.0], MLX::Core.float32))
          MLX::Core.eval(inner_value, inner_grad)
          inner_observation = [inner_value.item, inner_grad.to_a]
        end

        value = lifted.call(*flat_vars)
        grad = MLX::Core.multiply(flat_vars[0], MLX::Core.array(2.0, flat_vars[0].dtype))
        [value, grad]
      end
    end

    fn = lambda do |x|
      MLX::Core.sum(MLX::Core.multiply(x, x))
    end
    wrapped = MLX::Core.value_and_grad(fn)

    value, grad = wrapped.call(MLX::Core.array([2.0, -3.0], MLX::Core.float32))
    MLX::Core.eval(value, grad)

    assert_equal true, reentered
    refute_nil inner_observation
    assert_in_delta 26.0, inner_observation[0], 1e-5
    assert_equal [10.0, -2.0], inner_observation[1]
    assert_in_delta 13.0, value.item, 1e-5
    assert_equal [4.0, -6.0], grad.to_a
  ensure
    restore_singleton_override(:native_value_and_grad)
  end

  private

  def install_singleton_override(method_name, &block)
    singleton = class << MLX::Core
      self
    end
    backup = :"__phase327_backup_#{method_name}"
    singleton.alias_method(backup, method_name)
    singleton.remove_method(method_name) if singleton.instance_methods(false).include?(method_name)
    singleton.define_method(method_name, &block)
    @overrides ||= {}
    @overrides[method_name] = backup
  end

  def restore_singleton_override(method_name)
    return if @overrides.nil? || !@overrides.key?(method_name)

    singleton = class << MLX::Core
      self
    end
    backup = @overrides.delete(method_name)
    singleton.remove_method(method_name) if singleton.instance_methods(false).include?(method_name)
    singleton.alias_method(method_name, backup)
    singleton.remove_method(backup)
  end
end
