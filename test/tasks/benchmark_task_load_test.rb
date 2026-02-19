# frozen_string_literal: true

require_relative "../test_helper"

class BenchmarkTaskLoadTest < Minitest::Test
  def test_loading_benchmark_task_does_not_require_mlx
    script = <<~RUBY
      root = #{RUBY_ROOT.inspect}
      file = File.join(root, "tasks", "benchmark_task.rb")
      module Kernel
        alias_method :__benchmark_task_original_require, :require

        def require(name)
          if name == "mlx"
            raise LoadError, "mlx should not be required while loading benchmark_task.rb"
          end
          __benchmark_task_original_require(name)
        end
      end

      begin
        load file
        puts "ok"
      rescue LoadError => e
        warn e.message
        exit 1
      end
    RUBY

    stdout, stderr, status = Open3.capture3(RbConfig.ruby, "-e", script)

    assert status.success?, <<~MSG
      expected benchmark_task.rb to load without requiring mlx
      stdout:
      #{stdout}
      stderr:
      #{stderr}
    MSG
    assert_includes stdout, "ok"
  end
end
