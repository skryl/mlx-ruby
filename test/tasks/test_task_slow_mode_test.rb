# frozen_string_literal: true

require_relative "../support/test_helper"
require_relative "../../tasks/test_task"

class TestTaskSlowModeTest < Minitest::Test
  def setup
    @singleton = class << MlxTestTask
      self
    end
    @restores = []
    @had_include_slow = ENV.key?("MLX_TEST_INCLUDE_SLOW")
    @previous_include_slow = ENV["MLX_TEST_INCLUDE_SLOW"]
  end

  def teardown
    @restores.reverse_each do |name, backup|
      @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
      @singleton.alias_method(name, backup)
      @singleton.remove_method(backup)
    end

    if @had_include_slow
      ENV["MLX_TEST_INCLUDE_SLOW"] = @previous_include_slow
    else
      ENV.delete("MLX_TEST_INCLUDE_SLOW")
    end
  end

  def test_run_test_suite_for_devices_defaults_to_cpu_and_gpu_and_forwards_include_slow
    calls = []
    stub_singleton_method(:run_test_suite_for_device) do |device, include_slow:|
      calls << [device, include_slow]
    end

    MlxTestTask.run_test_suite_for_devices(nil, include_slow: true)

    assert_equal [["cpu", true], ["gpu", true]], calls
  end

  def test_with_include_slow_tests_sets_and_restores_environment_when_enabled
    ENV["MLX_TEST_INCLUDE_SLOW"] = "0"
    MlxTestTask.with_include_slow_tests(true) do
      assert_equal "1", ENV["MLX_TEST_INCLUDE_SLOW"]
    end
    assert_equal "0", ENV["MLX_TEST_INCLUDE_SLOW"]
  end

  def test_with_include_slow_tests_clears_and_restores_environment_when_disabled
    ENV["MLX_TEST_INCLUDE_SLOW"] = "1"
    MlxTestTask.with_include_slow_tests(false) do
      assert_nil ENV["MLX_TEST_INCLUDE_SLOW"]
    end
    assert_equal "1", ENV["MLX_TEST_INCLUDE_SLOW"]
  end

  private

  def stub_singleton_method(name, &block)
    backup = :"__test_task_slow_restore_#{name}_#{@restores.length}"
    @singleton.alias_method(backup, name)
    @restores << [name, backup]
    @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
    @singleton.define_method(name, &block)
  end
end
