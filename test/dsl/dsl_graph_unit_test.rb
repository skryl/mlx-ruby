# frozen_string_literal: true

require_relative "../test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class DslGraphUnitTest < Minitest::Test
  class AddOne < MLX::NN::Module
    def call(x)
      x + 1
    end
  end

  class MulTwo < MLX::NN::Module
    def call(x)
      x * 2
    end
  end

  class AddN < MLX::NN::Module
    def initialize(n)
      super()
      @n = n
    end

    def call(x)
      x + @n
    end
  end

  class SeedTuple < MLX::NN::Module
    def call(x, y:, scale: 1)
      [x + y, x - y, scale]
    end
  end

  class FoldTuple < MLX::NN::Module
    def call(a, b, scale)
      (a + b) * scale
    end
  end

  def with_core_stub(add_impl:, concatenate_impl:)
    singleton = class << MLX::Core
      self
    end
    methods = %i[add concatenate]
    restore = {}
    remove_stub = lambda do |name|
      next unless singleton.instance_methods(false).include?(name) ||
        singleton.private_instance_methods(false).include?(name) ||
        singleton.protected_instance_methods(false).include?(name)

      singleton.send(:remove_method, name)
    end

    methods.each do |name|
      backup = :"__dsl_graph_backup_#{name}"
      if singleton.private_instance_methods(false).include?(name)
        singleton.send(:alias_method, backup, name)
        remove_stub.call(name)
        restore[name] = :alias_private
      elsif singleton.protected_instance_methods(false).include?(name)
        singleton.send(:alias_method, backup, name)
        remove_stub.call(name)
        restore[name] = :alias_protected
      elsif singleton.instance_methods(false).include?(name)
        singleton.alias_method(backup, name)
        remove_stub.call(name)
        restore[name] = :alias_public
      else
        restore[name] = :remove
      end
    end

    singleton.define_method(:add, &add_impl)
    singleton.define_method(:concatenate, &concatenate_impl)
    yield
  ensure
    methods.each do |name|
      backup = :"__dsl_graph_backup_#{name}"
      remove_stub.call(name)
      next if restore[name] == :remove

      singleton.send(:alias_method, name, backup)
      singleton.send(:remove_method, backup)
      singleton.send(:private, name) if restore[name] == :alias_private
      singleton.send(:protected, name) if restore[name] == :alias_protected
    end
  end

  def test_parallel_returns_each_branch_output
    graph = MLX::DSL::Parallel.new(AddOne.new, MulTwo.new)
    out = graph.call(3)
    assert_equal [4, 6], out
  end

  def test_residual_uses_core_add
    with_core_stub(
      add_impl: ->(a, b) { a + b },
      concatenate_impl: ->(items, _axis = nil) { items }
    ) do
      graph = MLX::DSL::Residual.new(AddOne.new)
      out = graph.call(3)
      assert_equal 7, out
    end
  end

  def test_concat_uses_core_concatenate
    with_core_stub(
      add_impl: ->(a, b) { a + b },
      concatenate_impl: ->(items, _axis = nil) { items.flatten }
    ) do
      graph = MLX::DSL::Concat.new(
        AddOne.new,
        MulTwo.new,
        axis: -1
      )
      out = graph.call(3)
      assert_equal [4, 6], out
    end
  end

  def test_reduce_sum_uses_core_add
    with_core_stub(
      add_impl: ->(a, b) { a + b },
      concatenate_impl: ->(items, _axis = nil) { items }
    ) do
      graph = MLX::DSL::Reduce.new(AddOne.new, MulTwo.new, mode: :sum)
      out = graph.call(3)
      assert_equal 10, out
    end
  end

  def test_builder_sum_composition
    with_core_stub(
      add_impl: ->(a, b) { a + b },
      concatenate_impl: ->(items, _axis = nil) { items }
    ) do
      builder = MLX::DSL::Builder.new
      graph = builder.sum(AddOne.new, MulTwo.new)
      assert_instance_of MLX::DSL::Reduce, graph
      assert_equal 10, graph.call(3)
    end
  end

  def test_sequential_forwards_variadic_inputs_and_tuple_outputs
    seq = MLX::NN::Sequential.new(SeedTuple.new, FoldTuple.new)
    out = seq.call(4, y: 1, scale: 2)
    assert_equal 16, out
  end

  def test_builder_fn_wraps_inline_callable_layers
    builder = MLX::DSL::Builder.new
    layer = builder.fn { |x| x + 3 }
    assert_instance_of MLX::DSL::Callable, layer
    assert_equal 8, layer.call(5)
  end

  def test_builder_fn_forwards_variadic_args_and_kwargs
    builder = MLX::DSL::Builder.new
    layer = builder.fn { |x, y:, scale: 1| (x + y) * scale }
    assert_equal 14, layer.call(4, y: 3, scale: 2)
  end

  def test_builder_fn_requires_callable_or_block
    builder = MLX::DSL::Builder.new
    error = assert_raises(ArgumentError) { builder.fn(nil) }
    assert_match(/callable/i, error.message)
  end

  def test_builder_sequential_normalizes_module_classes_and_callables
    builder = MLX::DSL::Builder.new
    graph = builder.sequential(AddOne, ->(x) { x * 2 })
    assert_instance_of MLX::NN::Sequential, graph
    assert_instance_of AddOne, graph.layers[0]
    assert_instance_of MLX::DSL::Callable, graph.layers[1]
    assert_equal 8, graph.call(3)
  end

  def test_builder_branch_normalizes_module_classes_and_callables
    builder = MLX::DSL::Builder.new
    graph = builder.branch(AddOne, ->(x) { x * 2 })
    assert_instance_of MLX::DSL::Parallel, graph
    assert_instance_of AddOne, graph.layers[0]
    assert_instance_of MLX::DSL::Callable, graph.layers[1]
    assert_equal [4, 6], graph.call(3)
  end

  def test_builder_rejects_non_module_non_callable_entries
    builder = MLX::DSL::Builder.new
    error = assert_raises(TypeError) { builder.sequential(123) }
    assert_match(/MLX::NN::Module/, error.message)
  end

  def test_builder_layer_accepts_module_instance
    builder = MLX::DSL::Builder.new
    layer = builder.layer(AddOne.new)
    assert_instance_of AddOne, layer
    assert_equal 4, layer.call(3)
  end

  def test_builder_layer_accepts_module_class_with_constructor_args
    builder = MLX::DSL::Builder.new
    layer = builder.layer(AddN, 5)
    assert_instance_of AddN, layer
    assert_equal 8, layer.call(3)
  end

  def test_builder_layer_accepts_callable_and_block_form
    builder = MLX::DSL::Builder.new
    from_callable = builder.layer(->(x) { x * 3 })
    from_block = builder.layer { |x| x + 7 }

    assert_instance_of MLX::DSL::Callable, from_callable
    assert_instance_of MLX::DSL::Callable, from_block
    assert_equal 9, from_callable.call(3)
    assert_equal 10, from_block.call(3)
  end

  def test_builder_layer_rejects_args_for_module_instances_and_callables
    builder = MLX::DSL::Builder.new
    instance_error = assert_raises(ArgumentError) { builder.layer(AddOne.new, 1) }
    callable_error = assert_raises(ArgumentError) { builder.layer(->(x) { x }, foo: :bar) }

    assert_match(/does not accept constructor arguments/i, instance_error.message)
    assert_match(/does not accept constructor arguments/i, callable_error.message)
  end

  def test_builder_layer_requires_entry_or_block
    builder = MLX::DSL::Builder.new
    error = assert_raises(ArgumentError) { builder.layer }
    assert_match(/requires a module/i, error.message)
  end

  def test_builder_repeat_layers_supports_index_aware_block
    builder = MLX::DSL::Builder.new
    graph = builder.sequential do
      repeat_layers(3) do |index|
        fn { |x| x + index + 1 }
      end
    end

    assert_instance_of MLX::NN::Sequential, graph
    assert_equal 6, graph.call(0)
  end

  def test_builder_stack_repeats_layer_classes
    builder = MLX::DSL::Builder.new
    graph = builder.stack(3, AddN, 2)

    assert_instance_of MLX::NN::Sequential, graph
    assert_equal 9, graph.call(3)
    assert graph.layers.all? { |layer| layer.is_a?(AddN) }
  end

  def test_builder_stack_supports_index_aware_block_entries
    builder = MLX::DSL::Builder.new
    graph = builder.stack(3) do |index|
      ->(x) { x + index }
    end

    assert_instance_of MLX::NN::Sequential, graph
    assert_equal 6, graph.call(3)
    assert graph.layers.all? { |layer| layer.is_a?(MLX::DSL::Callable) }
  end
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
