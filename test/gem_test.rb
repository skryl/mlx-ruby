# frozen_string_literal: true

require_relative "test_helper"

class GemTest < Minitest::Test
  def setup
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_gem_loads
    assert_defined?(MLX)
    assert_equal "mlx", Gem::Specification.load(File.join(RUBY_ROOT, "mlx.gemspec")).name
    assert_match(/\A\d+\.\d+\.\d+(?:\.\d+)?\z/, MLX::VERSION)
  end

  def test_mlx_modules_available
    assert_const_defined(MLX, :Core)
    assert_const_defined(MLX, :Utils)
    assert_const_defined(MLX, :NN)
    assert_const_defined(MLX, :Optimizers)
    assert_const_defined(MLX, :DistributedUtils)
    assert_const_defined(MLX, :Extension)
    assert_const_defined(MLX, :DSL)

    assert_const_defined(MLX::NN, :Losses)
    assert_const_defined(MLX::NN, :Utils)
    assert_const_defined(MLX::NN, :Init)
    assert_const_defined(MLX::NN, :Module)
    assert_const_defined(MLX::NN, :Conv2d)
    assert_const_defined(MLX::NN, :Transformer)

    assert_const_defined(MLX::Optimizers, :Optimizer)
    assert_const_defined(MLX::Optimizers, :Schedulers)
    assert_respond_to(MLX::Optimizers::Schedulers, :exponential_decay)
    assert_respond_to(MLX::Optimizers::Schedulers, :step_decay)
    assert_respond_to(MLX::Optimizers::Schedulers, :cosine_decay)

    assert_const_defined(MLX::DistributedUtils, :Hostfile)
    assert_const_defined(MLX::DistributedUtils, :CommandProcess)
    assert_const_defined(MLX::DistributedUtils, :RemoteProcess)
    assert_const_defined(MLX::DistributedUtils, :IPConfigurator)
    assert_const_defined(MLX::DistributedUtils, :SSHInfo)
    assert_const_defined(MLX::DistributedUtils, :Host)
    assert_const_defined(MLX::DistributedUtils, :ThunderboltHost)
    assert_const_defined(MLX::DistributedUtils, :ThunderboltPort)

    assert_const_defined(MLX::Extension, :CMakeExtension)
    assert_const_defined(MLX::DSL, :Model)
    assert_const_defined(MLX::DSL, :ModelMixin)
    assert_const_defined(MLX::DSL, :Trainer)
  end

  private

  def assert_const_defined(mod, name)
    assert mod.const_defined?(name, false), "#{mod} should define constant #{name}"
  end

  def assert_defined?(value)
    assert value
  end
end
