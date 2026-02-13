# frozen_string_literal: true

require_relative "test_helper"

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

  def with_core_stub(add_impl:, concatenate_impl:)
    singleton = class << MLX::Core
      self
    end
    methods = %i[add concatenate]
    restore = {}

    methods.each do |name|
      backup = :"__dsl_graph_backup_#{name}"
      if singleton.instance_methods(false).include?(name)
        singleton.alias_method(backup, name)
        restore[name] = :alias
      elsif singleton.private_instance_methods(false).include?(name)
        singleton.send(:alias_method, backup, name)
        restore[name] = :alias_private
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
      case restore[name]
      when :alias
        singleton.alias_method(name, backup)
        singleton.remove_method(backup)
      when :alias_private
        singleton.send(:alias_method, name, backup)
        singleton.send(:remove_method, backup)
      when :remove
        singleton.remove_method(name) if singleton.instance_methods(false).include?(name)
      end
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
end

$LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
