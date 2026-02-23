# frozen_string_literal: true

require "rake"
require_relative "../test_helper"

class RakeTasksSmokeTest < Minitest::Test
  PROJECT_TASK_INVOCATIONS = [
    ["default"],
    ["test_base"],
    ["test"],
    ["test:fast"],
    ["test:all"],
    ["test:cpu"],
    ["test:gpu"],
    ["test:cpu_all"],
    ["test:gpu_all"],
    ["test:gem"],
    ["deps"],
    ["deps:ruby"],
    ["deps:python"],
    ["deps:web"],
    ["deps:all"],
    ["build"],
    ["docs:build"],
    ["gem:build"],
    ["gem:bump"],
    ["benchmark:deps"],
    ["benchmark:cpu"],
    ["benchmark:gpu"],
    ["benchmark:webgpu"],
    ["benchmark:graph_ir_coverage"],
    ["benchmark:all"],
    ["benchmark"],
    ["web:assets"],
    ["web:train"],
    ["web:start"],
    ["web:serve"]
  ].freeze

  def setup
    @restores = []
    @previous_rake_application = Rake.application
    @previous_strict_tests = ENV["MLX_STRICT_TESTS"]
    ENV["MLX_STRICT_TESTS"] = "1"

    Rake.application = Rake::Application.new
    load File.join(RUBY_ROOT, "Rakefile")
    stub_task_backends!
  end

  def teardown
    @restores.reverse_each do |singleton, name, backup|
      singleton.remove_method(name) if singleton.instance_methods(false).include?(name)
      singleton.alias_method(name, backup)
      singleton.remove_method(backup)
    end

    Rake.application = @previous_rake_application
    if @previous_strict_tests.nil?
      ENV.delete("MLX_STRICT_TESTS")
    else
      ENV["MLX_STRICT_TESTS"] = @previous_strict_tests
    end
  end

  def test_project_rake_tasks_invoke_without_error
    PROJECT_TASK_INVOCATIONS.each do |name, *args|
      assert Rake::Task.task_defined?(name), "expected rake task #{name} to be defined"

      task = Rake::Task[name]
      task.reenable
      task.invoke(*args)
    end
  end

  private

  def stub_task_backends!
    stub_singleton_method(MlxTestTask, :run_strict_test_suite!) { |*args, **kwargs| nil }
    stub_singleton_method(MlxTestTask, :run_test_suite_for_devices) { |*args, **kwargs| nil }
    stub_singleton_method(MlxTestTask, :run_test_suite_for_device) { |*args, **kwargs| nil }
    stub_singleton_method(MlxTestTask, :run_installed_gem_test_suite!) { |*args, **kwargs| nil }

    stub_singleton_method(DepsTask, :install_ruby_dependencies!) { |*args, **kwargs| nil }
    stub_singleton_method(DepsTask, :install_python_dependencies!) { |*args, **kwargs| nil }
    stub_singleton_method(DepsTask, :install_web_dependencies!) { |*args, **kwargs| nil }
    stub_singleton_method(DepsTask, :install_all!) { |*args, **kwargs| nil }

    stub_singleton_method(BuildTask, :build_native_extension!) { |*args, **kwargs| nil }
    stub_singleton_method(DocsTask, :build!) { |*args, **kwargs| nil }
    stub_singleton_method(GemTask, :build!) { |*args, **kwargs| nil }
    stub_singleton_method(GemTask, :bump_version!) { |*args, **kwargs| nil }

    stub_singleton_method(BenchmarkTask, :install_dependencies!) { |*args, **kwargs| nil }
    stub_singleton_method(BenchmarkTask, :run_cpu_task) { |*args, **kwargs| nil }
    stub_singleton_method(BenchmarkTask, :run_gpu_task) { |*args, **kwargs| nil }
    stub_singleton_method(BenchmarkTask, :run_webgpu_task) { |*args, **kwargs| nil }
    stub_singleton_method(BenchmarkTask, :run_graph_ir_coverage!) { |*args, **kwargs| nil }
    stub_singleton_method(BenchmarkTask, :run_all_task) { |*args, **kwargs| nil }
    stub_singleton_method(BenchmarkTask, :run_top_level_task) { |*args, **kwargs| nil }

    stub_singleton_method(WebAssetsTask, :run!) { |*args, **kwargs| nil }
    stub_singleton_method(TrainingTask, :run!) { |*args, **kwargs| nil }
    stub_singleton_method(WebTask, :start!) { |*args, **kwargs| nil }
  end

  def stub_singleton_method(owner, name, &block)
    singleton = class << owner
      self
    end
    backup = :"__rake_tasks_smoke_restore_#{name}_#{@restores.length}"
    singleton.alias_method(backup, name)
    @restores << [singleton, name, backup]
    singleton.remove_method(name) if singleton.instance_methods(false).include?(name)
    singleton.define_method(name, &block)
  end
end
