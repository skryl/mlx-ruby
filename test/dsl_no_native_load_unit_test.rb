# frozen_string_literal: true

require_relative "test_helper"

class DslNoNativeLoadUnitTest < Minitest::Test
  def test_model_mixin_loads_without_eager_core_dtype_resolution
    model_mixin = File.join(RUBY_ROOT, "lib", "mlx", "dsl", "model_mixin.rb")
    train_step = File.join(RUBY_ROOT, "lib", "mlx", "dsl", "train_step.rb")

    script = <<~RUBY
      module MLX
        module Core
        end

        module Optimizers
          class Optimizer
          end

          class MultiOptimizer
            def initialize(*)
            end
          end
        end

        module NN
          class Module
          end
        end

        module Utils
          def self.tree_flatten(*, destination: {})
            destination
          end
        end
      end

      require "#{train_step}"
      require "#{model_mixin}"

      class StubModule < MLX::NN::Module
        include MLX::DSL::ModelMixin
        param :weight, shape: [1]
      end

      puts "ok"
    RUBY

    stdout, stderr, status = Open3.capture3(RbConfig.ruby, "-e", script)
    assert status.success?, "subprocess failed:\nstdout=#{stdout}\nstderr=#{stderr}"
    assert_includes stdout, "ok"
  end
end
